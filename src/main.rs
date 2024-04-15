use std::{
    collections::HashMap,
    fs,
    path::PathBuf,
    str::FromStr,
    sync::Arc,
    time::{Duration, Instant},
};

use clap::{Parser, Subcommand};
use eyre::{bail, ContextCompat};
use ore::{
    state::{Bus, Proof, Treasury},
    utils::AccountDeserialize,
};
use serde_json::json;
use solana_client::{
    nonblocking::rpc_client::RpcClient,
    rpc_request::RpcRequest,
    rpc_response::{Response, RpcBlockhash},
};
use solana_sdk::{
    account::{Account, ReadableAccount},
    clock::{Clock, Slot},
    commitment_config::CommitmentConfig,
    keccak::Hash,
    pubkey::Pubkey,
    signature::{Keypair, Signature},
    signer::EncodableKey,
    sysvar,
};
use solana_transaction_status::TransactionStatus;
use tokio::io::AsyncWriteExt;
use tracing::{error, log};

mod batch_transfer;
mod benchmark_rpc;
mod bundle_mine;
mod bundle_mine_gpu;
mod claim;
mod constant;
mod generate_wallet;
mod register;
mod utils;

#[tokio::main(flavor = "multi_thread")]
async fn main() {
    Miner::init_pretty_env_logger();
    let miner = Miner::parse();

    match &miner.command {
        Command::Claim(args) => miner.claim(args).await,
        Command::BundleMine(args) => miner.bundle_mine(args).await,
        Command::BundleMineGpu(args) => miner.bundle_mine_gpu(args).await,
        Command::Register(args) => miner.register(args).await,
        Command::BenchmarkRpc(args) => miner.benchmark_rpc(args).await,
        Command::BatchTransfer(args) => miner.batch_transfer(args).await,
        Command::GenerateWallet(args) => miner.generate_wallet(args),
    }
}

#[derive(Parser, Debug, Clone)]
pub struct Miner {
    #[arg(long, default_value = "https://api.mainnet-beta.solana.com")]
    pub rpc: String,

    #[arg(long)]
    pub priority_fee: Option<u64>,

    #[command(subcommand)]
    pub command: Command,
}

#[derive(Subcommand, Debug, Clone)]
pub enum Command {
    Claim(crate::claim::ClaimArgs),
    BundleMine(crate::bundle_mine::BundleMineArgs),
    BundleMineGpu(crate::bundle_mine_gpu::BundleMineGpuArgs),
    Register(crate::register::RegisterArgs),
    BenchmarkRpc(crate::benchmark_rpc::BenchmarkRpcArgs),
    GenerateWallet(crate::generate_wallet::GenerateWalletArgs),
    BatchTransfer(crate::batch_transfer::BatchTransferArgs),
}

impl Miner {
    pub fn init_pretty_env_logger() {
        env_logger::Builder::new()
            .filter_level(log::LevelFilter::Info)
            .parse_default_env()
            .init();
    }

    pub fn get_client_confirmed(rpc: &str) -> Arc<RpcClient> {
        Arc::new(RpcClient::new_with_commitment(
            rpc.to_string(),
            CommitmentConfig::confirmed(),
        ))
    }

    pub fn read_keys(key_folder: &str) -> Vec<Keypair> {
        fs::read_dir(key_folder)
            .expect("Failed to read key folder")
            .map(|entry| {
                let path = entry.expect("Failed to read entry").path();

                Keypair::read_from_file(&path).unwrap_or_else(|_| panic!("Failed to read keypair from {:?}", path))
            })
            .collect::<Vec<_>>()
    }

    pub async fn get_latest_blockhash_and_slot(client: &RpcClient) -> eyre::Result<(Slot, solana_sdk::hash::Hash)> {
        let (blockhash, send_at_slot) = match client
            .send::<Response<RpcBlockhash>>(RpcRequest::GetLatestBlockhash, json!([{"commitment": "confirmed"}]))
            .await
        {
            Ok(r) => (r.value.blockhash, r.context.slot),
            Err(err) => eyre::bail!("failed to get latest blockhash: {err:#}"),
        };

        let blockhash = match solana_sdk::hash::Hash::from_str(&blockhash) {
            Ok(b) => b,
            Err(err) => eyre::bail!("fail to parse blockhash: {err:#}"),
        };

        Ok((send_at_slot, blockhash))
    }

    pub async fn mine_hashes_cpu(
        &self,
        threads: usize,
        difficulty: &Hash,
        hash_and_pubkey: &[(Hash, Pubkey)],
    ) -> (Duration, Vec<(Hash, u64)>) {
        self.mine_hashes(utils::get_nonce_worker_path(), threads, difficulty, hash_and_pubkey)
            .await
    }

    pub async fn mine_hashes_gpu(
        &self,
        difficulty: &Hash,
        hash_and_pubkey: &[(Hash, Pubkey)],
    ) -> (Duration, Vec<(Hash, u64)>) {
        self.mine_hashes(utils::get_gpu_nonce_worker_path(), 0, difficulty, hash_and_pubkey)
            .await
    }

    pub async fn mine_hashes(
        &self,
        worker: PathBuf,
        threads: usize,
        difficulty: &Hash,
        hash_and_pubkey: &[(Hash, Pubkey)],
    ) -> (Duration, Vec<(Hash, u64)>) {
        let mining_start = Instant::now();

        let mut child = tokio::process::Command::new(worker)
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .spawn()
            .expect("nonce_worker failed to spawn");

        {
            let stdin = child.stdin.as_mut().unwrap();

            stdin.write_u8(threads as u8).await.unwrap();
            stdin.write_all(difficulty.as_ref()).await.unwrap();

            for (hash, pubkey) in hash_and_pubkey {
                stdin.write_all(hash.as_ref()).await.unwrap();
                stdin.write_all(pubkey.as_ref()).await.unwrap();
            }
        }

        let output = child.wait_with_output().await.unwrap().stdout;
        let mut results = vec![];

        for item in output.chunks(40) {
            let hash = Hash(item[..32].try_into().unwrap());
            let nonce = u64::from_le_bytes(item[32..40].try_into().unwrap());

            results.push((hash, nonce));
        }

        let mining_duration = mining_start.elapsed();
        (mining_duration, results)
    }

    pub fn find_buses(buses: [Bus; ore::BUS_COUNT], required_reward: u64) -> Vec<Bus> {
        let mut available_bus = buses
            .into_iter()
            .filter(|bus| bus.rewards >= required_reward)
            .collect::<Vec<_>>();

        available_bus.sort_by(|a, b| b.rewards.cmp(&a.rewards));

        available_bus
    }

    pub async fn get_accounts(
        id: usize,
        client: &RpcClient,
        accounts: &[Pubkey],
    ) -> Option<(Treasury, Clock, [Bus; ore::BUS_COUNT])> {
        let now = Instant::now();
        let mut result = None;

        let mut account_infos = client
            .get_multiple_accounts_with_commitment(accounts, CommitmentConfig::confirmed())
            .await
            .unwrap_or_else(|err| panic!("Failed to get multiple accounts: {}", err));

        let clock = match Clock::from_account(&account_infos[id].1) {
            Ok(c) => c,
            Err(_) => return None,
        };

        let treasury = match Treasury::from_account(&account_infos[id].1) {
            Ok(t) => t,
            Err(_) => return None,
        };

        let mut buses = [Bus::default(); ore::BUS_COUNT];
        for (info, account_id) in account_infos.iter_mut().zip(accounts.iter()) {
            if let Ok(bus) = Bus::from_account(&info.1) {
                buses[account_id.index()] = bus;
            }
        }

        result = Some((treasury, clock, buses));
        log::info!("get_accounts {}ms", now.elapsed().as_millis());
        result
    }

    pub async fn send_and_confirm_transaction(
        client: &RpcClient,
        transaction: &solana_sdk::transaction::Transaction,
        payer: &Keypair,
    ) -> eyre::Result<Signature> {
        const SEND_RETRIES: usize = 5;
        const CONFIRM_RETRIES: usize = 5;
        const POLL_DELAY: Duration = Duration::from_millis(10);

        for _ in 0..SEND_RETRIES {
            let signature = client.send_transaction(transaction).await?;

            for _ in 0..CONFIRM_RETRIES {
                if let Some(result) = client.get_signature_status_with_commitment(&signature, CommitmentConfig::confirmed()).await? {
                    match result {
                        Ok(_) => return Ok(signature),
                        Err(_) => break,
                    }
                }
                tokio::time::sleep(POLL_DELAY).await;
            }
        }

        bail!("send_and_confirm_transaction failed")
    }
}
