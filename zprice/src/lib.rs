// Copyright (c) 2022 Espresso Systems (espressosys.com)
// This file is part of the Jellyfish library.

// You should have received a copy of the MIT License
// along with the Jellyfish library. If not, see <https://mit-license.org/>.

//! Helper functions and testing/bechmark code for ZPrice: Plonk-DIZK GPU
//! acceleration

#[cfg(test)]
extern crate std;

use jf_utils::Vec;

use ark_ed_on_bls12_381::{EdwardsParameters, Fq};
use ark_std::rand::Rng;
use jf_plonk::prelude::*;
use jf_primitives::{
    circuit::signature::schnorr::SignatureGadget, constants::CS_ID_SCHNORR,
    signatures::schnorr::KeyPair,
};

const NUM_SIGS: u64 = 200;

pub fn generate_circuit<R: Rng>(rng: &mut R) -> Result<PlonkCircuit<Fq>, PlonkError> {
    let mut circuit = PlonkCircuit::new();

    for _ in 0..NUM_SIGS {
        let keypair = KeyPair::<EdwardsParameters>::generate(rng);
        let vk = keypair.ver_key_ref();
        let msg = (0..20).map(|i| Fq::from(i as u64)).collect::<Vec<_>>();
        let sig = keypair.sign(&msg, CS_ID_SCHNORR);
        vk.verify(&msg, &sig, CS_ID_SCHNORR).unwrap();

        let vk_var = circuit.create_signature_vk_variable(vk)?;
        let sig_var = circuit.create_signature_variable(&sig)?;
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
    println!("{}", circuit.num_gates());

    // sanity check: the circuit must be satisfied.
    assert!(circuit.check_circuit_satisfiability(&[]).is_ok());
    circuit.finalize_for_arithmetization()?;

    Ok(circuit)
}

#[cfg(test)]
mod tests {
    use std::{println, time::Instant};

    use ark_bls12_381::Bls12_381;
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
        println!("{:?}", now.elapsed());
        let now = Instant::now();
        sigs.par_iter().for_each(|sig| {
            vk.verify(&msg, &sig, CS_ID_SCHNORR).unwrap();
        });
        println!("{:?}", now.elapsed());
    }

    #[test]
    fn test_cpu() {
        let rng = &mut rand::thread_rng();

        let circuit = generate_circuit(rng).unwrap();

        let max_degree = circuit.srs_size().unwrap();

        let srs = PlonkKzgSnark::<Bls12_381>::universal_setup(max_degree, rng).unwrap();
        let (pk, vk) = PlonkKzgSnark::<Bls12_381>::preprocess(&srs, &circuit).unwrap();

        let now = Instant::now();
        let proof =
            PlonkKzgSnark::<Bls12_381>::prove::<_, _, StandardTranscript>(rng, &circuit, &pk)
                .unwrap();
        println!("{:?}", now.elapsed());
        let public_inputs = circuit.public_input().unwrap();
        let now = Instant::now();
        assert!(PlonkKzgSnark::<Bls12_381>::verify::<StandardTranscript>(
            &vk,
            &public_inputs,
            &proof,
        )
        .is_ok());
        println!("{:?}", now.elapsed());
    }

    #[test]
    fn test_gpu() {
        let rng = &mut rand::thread_rng();

        let circuit = generate_circuit(rng).unwrap();

        let max_degree = circuit.srs_size().unwrap();

        let srs = PlonkKzgSnark::<Bls12_381>::universal_setup(max_degree, rng).unwrap();
        let (pk, vk) = PlonkKzgSnark::<Bls12_381>::preprocess(&srs, &circuit).unwrap();

        let now = Instant::now();
        let proof = prover_single_gpu::Prover::prove(rng, &circuit, &pk).unwrap();
        println!("{:?}", now.elapsed());
        let public_inputs = circuit.public_input().unwrap();
        assert!(PlonkKzgSnark::<Bls12_381>::verify::<StandardTranscript>(
            &vk,
            &public_inputs,
            &proof,
        )
        .is_ok());
    }
}
