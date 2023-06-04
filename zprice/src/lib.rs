// Copyright (c) 2022 Espresso Systems (espressosys.com)
// This file is part of the Jellyfish library.

// You should have received a copy of the MIT License
// along with the Jellyfish library. If not, see <https://mit-license.org/>.

//! Helper functions and testing/bechmark code for ZPrice: Plonk-DIZK GPU
//! acceleration

#[cfg(test)]
extern crate std;

use std::{path::PathBuf, io::BufReader};

use ark_serialize::{CanonicalDeserialize, CanonicalSerialize, Write, Read};
use jf_utils::Vec;

use ark_ed_on_bls12_381::{EdwardsParameters, Fq};
use ark_std::rand::Rng;
use jf_plonk::prelude::*;
use jf_primitives::{
    circuit::signature::schnorr::SignatureGadget, constants::CS_ID_SCHNORR,
    signatures::schnorr::{KeyPair, Signature, VerKey},
};

const NUM_SIGS: u64 = 50;

pub fn generate_circuit<R: Rng>(rng: &mut R) -> Result<PlonkCircuit<Fq>, PlonkError> {
    let mut circuit = PlonkCircuit::new();

    let vk_path = default_path("vk", "bin");
    // println!("loading signature verifying key from: {}", vk_path.to_str().unwrap());
    let vk: VerKey<EdwardsParameters> = load_data(vk_path);

    let msg_path = default_path("msg", "bin");
    // println!("loading messages from: {}", msg_path.to_str().unwrap());
    let msg: Vec<Fq> = load_data(msg_path);

    let sigs_path = default_path("sig", "bin");
    // println!("loading signatures from: {}", sigs_path.to_str().unwrap());
    let sigs: Vec<Signature<EdwardsParameters>> = load_data(sigs_path);

    for i in 0..NUM_SIGS {
        // let keypair = KeyPair::<EdwardsParameters>::generate(rng);
        // let vk = keypair.ver_key_ref();
        // let msg = (0..20).map(|i| Fq::from(i as u64)).collect::<Vec<_>>();
        // let sig = keypair.sign(&msg, CS_ID_SCHNORR);
        vk.verify(&msg, &sigs[i as usize], CS_ID_SCHNORR).unwrap();

        let vk_var = circuit.create_signature_vk_variable(&vk)?;
        let sig_var = circuit.create_signature_variable(&sigs[i as usize])?;
        let msg_var = msg
            .iter()
            .map(|m| circuit.create_variable(*m))
            .collect::<Result<Vec<_>, PlonkError>>()?;
        SignatureGadget::<Fq, EdwardsParameters>::verify_signature(
            &mut circuit,
            &vk_var,
            &msg_var,
            &sig_var,
        )?;
    }

    // sanity check: the circuit must be satisfied.
    assert!(circuit.check_circuit_satisfiability(&[]).is_ok());
    circuit.finalize_for_arithmetization()?;

    Ok(circuit)
}

fn default_path(filename: &str, extension: &str) -> PathBuf {
    let mut d = PathBuf::from(ark_std::env::var("CARGO_MANIFEST_DIR").unwrap());
    d.push("data");
    d.push(filename);
    d.set_extension(extension);
    d
}

fn store_data<T>(data: &T, dest: PathBuf)
where
    T: CanonicalSerialize,
{
    let mut bytes = Vec::new();
    data.serialize_unchecked(&mut bytes).unwrap();
    store_bytes(&bytes, dest);
}

// deserialize any serde-deserializable data using `bincode` from `src`
fn load_data<T>(src: PathBuf) -> T
where
    T: CanonicalDeserialize,
{
    let bytes = load_bytes(src);
    T::deserialize_unchecked(&bytes[..]).unwrap()
}

fn store_bytes(bytes: &[u8], dest: PathBuf) {
    let mut f = ark_std::fs::File::create(dest).unwrap();
    f.write_all(bytes).unwrap();
}

fn load_bytes(src: PathBuf) -> Vec<u8> {
    let f = ark_std::fs::File::open(src).unwrap();
    let mut reader = BufReader::new(f);
    let mut bytes = Vec::new();
    reader.read_to_end(&mut bytes).unwrap();
    bytes
}
#[cfg(test)]
mod tests {
    use std::{println, time::Instant};

    use ark_bls12_381::{Bls12_381, G1Affine, Fr as Fr381};
    use ark_serialize::CanonicalSerialize;
    use jf_primitives::signatures::schnorr::{VerKey, Signature};
    use rayon::prelude::{IntoParallelRefIterator, ParallelIterator, IntoParallelIterator};

    use super::*;

    #[test]
    fn test_sig() {
        let rng = &mut rand::thread_rng();
        let keypair = KeyPair::<EdwardsParameters>::generate(rng);
        let vk = keypair.ver_key_ref();
        let msg = (0..20).map(|i| Fq::from(i as u64)).collect::<Vec<_>>();
        let now = Instant::now();
        let sigs = (0..NUM_SIGS)
            .into_par_iter()
            .map(|_| keypair.sign(&msg, CS_ID_SCHNORR))
            .collect::<Vec<_>>();
        println!("sign: {:?}", now.elapsed());
        let now = Instant::now();
        sigs.par_iter().for_each(|sig| {
            vk.verify(&msg, &sig, CS_ID_SCHNORR).unwrap();
        });
        println!("verify sig: {:?}", now.elapsed());
    }

    #[test]
    fn gen_srs_keys(){
        let rng = &mut rand::thread_rng();
        let circuit: PlonkCircuit<ark_ff::Fp256<ark_bls12_381::FrParameters>> = generate_circuit(rng).unwrap();
        let max_degree = circuit.srs_size().unwrap();
        println!("generate srs..");
        let srs = PlonkKzgSnark::<Bls12_381>::universal_setup(max_degree, rng).unwrap();
        println!("generate proving key and verifying key..");
        let (pk, vk) = PlonkKzgSnark::<Bls12_381>::preprocess(&srs, &circuit).unwrap();

        let pk_path = default_path("pk", "bin");
        store_data(&pk, pk_path.clone());
        println!("storing proving key to: {}", pk_path.to_str().unwrap());

        let vk_inputs_path = default_path("zk_vk", "bin");
        store_data(&vk, vk_inputs_path.clone());
        println!("storing zk verifying key to: {}", vk_inputs_path.to_str().unwrap());

        let srs_path = default_path("srs", "bin");
        store_data(&srs, srs_path.clone());
        println!("storing srs to: {}", srs_path.to_str().unwrap());
    }

    #[test]
    fn gen_cpu() {
        let rng = &mut rand::thread_rng();

        let pk_path = default_path("pk", "bin");
        println!("loading zk proving key from: {}", pk_path.to_str().unwrap());
        let pk = load_data(pk_path);

        println!("generate circuit..");
        let circuit = generate_circuit(rng).unwrap();
        println!("num of gates: {}", circuit.num_gates());

        let now = Instant::now();
        let proof: Proof<ark_ec::bls12::Bls12<ark_bls12_381::Parameters>> =
            PlonkKzgSnark::<Bls12_381>::prove::<_, _, StandardTranscript>(rng, &circuit, &pk)
                .unwrap();
        println!("CPU version proving time: {:?}", now.elapsed());
        println!("Proof size: {:?} bytes", proof.serialized_size());

        let public_inputs = circuit.public_input().unwrap();

        let public_inputs_path = default_path("public_inputs", "bin");
        store_data(&public_inputs, public_inputs_path.clone());
        println!("storing public input to: {}", public_inputs_path.to_str().unwrap());
        
        let proof_path = default_path("proof", "bin");
        store_data(&proof, proof_path.clone());
        println!("storing proof to: {}", proof_path.to_str().unwrap());

    }

    #[test]
    fn gen_gpu() {
        let rng = &mut rand::thread_rng();

        let pk_path = default_path("pk", "bin");
        println!("loading zk proving key from: {}", pk_path.to_str().unwrap());
        let pk = load_data(pk_path);

        println!("generate circuit..");
        let circuit = generate_circuit(rng).unwrap();
        println!("num of gates: {}", circuit.num_gates());

        let now = Instant::now();
        let proof = prover_single_gpu::Prover::prove(rng, &circuit, &pk).unwrap();
        println!("GPU version proving time: {:?}", now.elapsed());
        println!("proof size: {:?}", proof.serialized_size());
        let public_inputs = circuit.public_input().unwrap();

        let public_inputs_path = default_path("public_inputs", "bin");
        store_data(&public_inputs, public_inputs_path.clone());
        println!("storing public input to: {}", public_inputs_path.to_str().unwrap());

        let proof_path = default_path("proof", "bin");
        store_data(&proof, proof_path.clone());
        println!("storing proof to: {}", proof_path.to_str().unwrap());

    }

    #[test]
    fn verify_proof() {
        let proof_path = default_path("proof", "bin");
        println!("loading proof from: {}", proof_path.to_str().unwrap());
        let proof = load_data(proof_path);

        let public_inputs_path = default_path("public_inputs", "bin");
        println!("loading public inputs from: {}", public_inputs_path.to_str().unwrap());
        let public_inputs:Vec<Fr381> = load_data(public_inputs_path);

        let vk_inputs_path = default_path("zk_vk", "bin");
        println!("loading zk verifying key from: {}", vk_inputs_path.to_str().unwrap());
        let vk = load_data(vk_inputs_path);
        
        let now = Instant::now();
        assert!(PlonkKzgSnark::<Bls12_381>::verify::<StandardTranscript>(
            &vk,
            &public_inputs,
            &proof,
        )
        .is_ok());
        println!("proof verify time: {:?}", now.elapsed());
    }
    #[test]
    fn gen_sig() {
        let rng = &mut rand::thread_rng();
        let keypair = KeyPair::<EdwardsParameters>::generate(rng);
        let vk = keypair.ver_key_ref();
        let msg = (0..20).map(|i| Fq::from(i as u64)).collect::<Vec<_>>();
        let now = Instant::now();
        let sigs = (0..NUM_SIGS)
            .into_par_iter()
            .map(|_| keypair.sign(&msg, CS_ID_SCHNORR))
            .collect::<Vec<_>>();
        println!("signing time: {:?}", now.elapsed());
        let vk_path = default_path("vk", "bin");
        store_data(vk, vk_path.clone());
        println!("storing verifying key to: {}", vk_path.to_str().unwrap());

        let msg_path = default_path("msg", "bin");
        store_data(&msg, msg_path.clone());
        println!("storing message to: {}", msg_path.to_str().unwrap());

        let sig_path = default_path("sig", "bin");
        store_data(&sigs, sig_path.clone());
        println!("storing signatures to: {}", sig_path.to_str().unwrap());
    }

    #[test]
    fn verify_sig() {
        let vk_path = default_path("vk", "bin");
        println!("loading verifying key from: {}", vk_path.to_str().unwrap());
        let vk: VerKey<EdwardsParameters> = load_data(vk_path);

        let msg_path = default_path("msg", "bin");
        println!("loading messages from: {}", msg_path.to_str().unwrap());
        let msg: Vec<Fq> = load_data(msg_path);

        let sigs_path = default_path("sig", "bin");
        println!("loading signatures from: {}", sigs_path.to_str().unwrap());
        let sigs: Vec<Signature<EdwardsParameters>> = load_data(sigs_path);

        let now = Instant::now();
        sigs.par_iter().for_each(|sig| {
            vk.verify(&msg, &sig, CS_ID_SCHNORR).unwrap();
        });
        println!("verify sig time: {:?}", now.elapsed());
    }
}
