version 1.0


workflow scbtf_workflow {
	input {
		File adata_path
		Array[Int] ranks
		String sample_label
		String celltype_label
		String output_directory
		String method = "BTF"
		Int n_restarts = 50
		Int scale_to = 1000000
		Int filter_gene_count = 50
		Boolean hgnc_approved_genes_only = true
		Boolean normalize = true
		Boolean variance_scale = true
		Boolean sqrt_transform = true
		String model = "gamma_poisson"
		Int num_steps = 1500
		Float initial_lr = 1.0
		Float lr_decay_gamma = 0.001
		Float init_alpha = 100.0

		Int preemptible = 0
		Int disk_space = 50
		Int num_cpu = 16
		String memory = "120G"
		String docker_version = "scbtf:0.0.1"
		String zones = "us-east1-d us-west1-a us-west1-b"
	}

	call tensorize  {
		input:
			adata_path = adata_path,
			sample_label = sample_label,
			celltype_label = celltype_label,
			scale_to = scale_to,
			filter_gene_count = filter_gene_count,
			hgnc_approved_genes_only = hgnc_approved_genes_only,
			normalize = normalize,
			variance_scale = variance_scale,
			sqrt_transform = sqrt_transform,

			preemptible = preemptible,
			disk_space = disk_space,
			docker_version = docker_version,
			num_cpu = num_cpu,
			memory = memory,
			zones = zones
	}

	scatter(rank in ranks) {
		call factorize_rank as scattered_factorize_rank {
			input:
				tensor_path = tensorize.tensor,
				rank = rank,
				method = method,
				model = model,
				n_restarts = n_restarts,
				num_steps = num_steps,
				initial_lr = initial_lr,
				lr_decay_gamma = lr_decay_gamma,
				init_alpha = init_alpha,

				preemptible = preemptible,
				disk_space = disk_space,
				docker_version = docker_version,
				num_cpu = num_cpu,
				memory = memory,
				zones = zones
		}
	}

	call aggregate_factorization  {
		input:
			consensus_factorizations = scattered_factorize_rank.consensus_factorization,
			full_factorizations = scattered_factorize_rank.full_factorization,
			output_directory = output_directory,

			preemptible = preemptible,
			disk_space = disk_space,
			docker_version = docker_version,
			num_cpu = num_cpu,
			memory = memory,
			zones = zones
	}

	output {
		String output_dir = output_directory
	}

}


task tensorize {

	input {
		File adata_path
		String sample_label
		String celltype_label
		Int scale_to
		Int filter_gene_count
		Boolean hgnc_approved_genes_only
		Boolean normalize
		Boolean variance_scale
		Boolean sqrt_transform

		Int preemptible
		Int disk_space
		Int num_cpu
		String memory
		String docker_version
		String zones
	}

	command <<<
		set -e
		mkdir -p results
		source ~/.bashrc

		python <<CODE

		import pickle
		import scanpy as sc
		from rich import print
		from scBTF import SingleCellTensor

		adata = sc.read('~{adata_path}')
		print(adata.shape)

		sc_tensor = SingleCellTensor.from_anndata(
			adata, sample_label='~{sample_label}', celltype_label='~{celltype_label}', scale_to=~{scale_to},
			normalize=('~{normalize}'=='true'), variance_scale=('~{variance_scale}'=='true'),
			sqrt_transform=('~{sqrt_transform}'=='true'), hgnc_approved_genes_only=('~{hgnc_approved_genes_only}'=='true'),
			filter_gene_count=~{filter_gene_count}
		)

		with open('results/tensor.pkl', 'wb') as file:
			pickle.dump(sc_tensor, file)
		print("Saved tensor of shape : ", sc_tensor.tensor.shape)

		CODE
	>>>

	output {
		File tensor = "results/tensor.pkl"
	}

	runtime {
		preemptible: preemptible
		bootDiskSizeGb: 12
		disks: "local-disk ${disk_space} HDD"
		docker: "gcr.io/microbiome-xavier/${docker_version}"
		cpu: num_cpu
		zones: zones
		memory: memory
	}


}


task factorize_rank {

	input {
		File tensor_path
		Int rank
		String method
		String model
		Int n_restarts
		Int num_steps
		Float initial_lr
		Float lr_decay_gamma
		Float init_alpha

		Int preemptible
		Int disk_space
		Int num_cpu
		String memory
		String docker_version
		String zones
	}

	command <<<
		set -e
		mkdir -p results
		source ~/.bashrc

		python <<CODE

		import pickle
		import torch
		import numpy as np
		from rich import print
		from scBTF import SingleCellBTF, FactorizationSet

		with open('~{tensor_path}', 'rb') as file:
			sc_tensor = pickle.load(file)
		print("Loaded tensor of shape : ", sc_tensor.tensor.shape)

		if '~{method}' == "BTF":
			sc_tensor.tensor = (sc_tensor.tensor*1e5/sc_tensor.tensor.max()).round()
			factorization_set = SingleCellBTF.factorize(
				sc_tensor=sc_tensor,
				rank=~{rank},
				model='~{model}',
				n_restarts=~{n_restarts},
				num_steps=~{num_steps},
				init_alpha=~{init_alpha},
				initial_lr=~{initial_lr},
				lr_decay_gamma=~{lr_decay_gamma},
				plot_var_explained=False
			)
		elif '~{method}' == "HALS":
			factorization_set = SingleCellBTF.factorize_hals(
				sc_tensor=sc_tensor,
				rank=~{rank},
				num_steps=~{num_steps},
				n_restarts=~{n_restarts},
				sparsity_coefficients=[0., 0., 10.],
				plot_var_explained=False
			)

		factorization_set.save("results/rank_~{rank}_factorization.pkl")

		reconstructed_all = FactorizationSet()
		reconstructed_all.sc_tensor = factorization_set.sc_tensor
		reconstructed_all.sc_tensor.tensor = (reconstructed_all.sc_tensor.tensor*1e5/reconstructed_all.sc_tensor.tensor.max()).round()

		for selected_rank in factorization_set.get_ranks():
			# Use median of clustered gene factors to reconstruct a final factorization
			medians = factorization_set.reconstruct_factors_from_median_gene_factor(rank = selected_rank)
			reconstructed = SingleCellBTF.factorize(
				sc_tensor = factorization_set.sc_tensor,
				rank = selected_rank,
				n_restarts = 1,
				init_alpha = ~{init_alpha},
				num_steps = ~{num_steps},
				initial_lr=~{initial_lr},
				lr_decay_gamma=~{lr_decay_gamma},
				model = 'gamma_poisson_fixed',
				fixed_mode = 2,
				fixed_value = torch.from_numpy(medians.T).float(),
				plot_var_explained = False
			)

			gene_factor = reconstructed.get_factorization(rank = selected_rank, restart_index = 0).gene_factor['mean'].numpy()
			print((1 - np.isclose(medians.T, gene_factor, atol=2)).sum(), '/', medians.flatten().shape[0], ' mismatches in final gene factors')
			print('variance explained by reconstructed factorization = ', reconstructed.variance_explained(rank=selected_rank, restart_index=0).item())
			reconstructed_all.factorizations[selected_rank] = reconstructed.factorizations[selected_rank]
		reconstructed_all.save('results/rank_~{rank}_consensus_factorization.pkl')

		CODE
	>>>

	output {
		File full_factorization = "results/rank_~{rank}_factorization.pkl"
		File consensus_factorization = "results/rank_~{rank}_consensus_factorization.pkl"
	}

	runtime {
		preemptible: preemptible
		bootDiskSizeGb: 12
		disks: "local-disk ${disk_space} HDD"
		docker: "gcr.io/microbiome-xavier/${docker_version}"
		cpu: num_cpu
		zones: zones
		memory: memory
	}


}


task aggregate_factorization {

	input {
		Array[File] consensus_factorizations
		Array[File] full_factorizations
		String output_directory

		Int preemptible
		Int disk_space
		Int num_cpu
		String memory
		String docker_version
		String zones
	}

	command <<<
		set -e
		mkdir -p results
		source ~/.bashrc

		python <<CODE

		from scBTF import FactorizationSet

		reconstructed_all = FactorizationSet()
		for consensus_path in '~{sep="," consensus_factorizations}'.split(','):
			reconstructed = FactorizationSet.load(consensus_path)
			rank = list(reconstructed.get_ranks())[0]
			reconstructed_all.factorizations[rank] = reconstructed.factorizations[rank]
		reconstructed_all.sc_tensor = reconstructed.sc_tensor
		reconstructed_all.save('results/consensus_factorization.pkl')

		full_factorization_all = FactorizationSet()
		for full_factorization_path in '~{sep="," full_factorizations}'.split(','):
			full_factorization = FactorizationSet.load(full_factorization_path)
			rank = list(full_factorization.get_ranks())[0]
			full_factorization_all.factorizations[rank] = full_factorization.factorizations[rank]
		full_factorization_all.sc_tensor = full_factorization.sc_tensor
		full_factorization_all.save('results/full_factorization.pkl')

		CODE

		gsutil -m cp -r "results/" ~{output_directory}/

	>>>

	output {
		File all_consensus_factorizations = 'results/consensus_factorization.pkl'
		File all_full_factorizations = 'results/full_factorization.pkl'
	}

	runtime {
		preemptible: preemptible
		bootDiskSizeGb: 12
		disks: "local-disk ${disk_space} HDD"
		docker: "gcr.io/microbiome-xavier/${docker_version}"
		cpu: num_cpu
		zones: zones
		memory: memory
	}
}

task generate_summary {

	input {
		File consensus_factorization
		File full_factorization
		File adata_path
		String output_directory

		Int preemptible
		Int disk_space
		Int num_cpu
		String memory
		String docker_version
		String zones
	}

	command <<<
		set -e
		mkdir -p results
		source ~/.bashrc

		python <<CODE

		import os

		ARGS = 'stub.py --adata_path {} --consensus_factorization_path {} --full_factorization_path {}'
		CONFIG_FILENAME = '.config_ipynb'

		with open(CONFIG_FILENAME, 'w') as f:
			f.write(ARGS.format('~{adata_path}', '~{consensus_factorization}', '~{full_factorization}'))

		with open('.script.sh', 'w') as f:
			f.write('#!/bin/bash\n')
			f.write('jupyter nbconvert --execute rank_determination_template.ipynb --output factor_analysis')
		os.system('bash .script.sh')

		CODE

		gsutil -m cp 'factor_analysis.ipynb' '~{output_directory}/results/'

	>>>

	runtime {
		preemptible: preemptible
		bootDiskSizeGb: 12
		disks: "local-disk ${disk_space} HDD"
		docker: "gcr.io/microbiome-xavier/${docker_version}"
		cpu: num_cpu
		zones: zones
		memory: memory
	}
}