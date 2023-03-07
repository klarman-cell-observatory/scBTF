version 1.0

## Version 3-3-2023
##
## Distributed under terms of the MIT License
## Copyright (c) 2023 Daniel Chafamo
## Contact <chafamodaniel@gmail.com>

workflow scbtf_workflow {

	input {
		File adata_path
		Array[Int] ranks
		String sample_label
		String celltype_label
		String output_directory
		Int n_restarts = 50
		Int scale_to = 1000000
		Int filter_gene_count = 50
		Boolean hgnc_approved_genes_only = True
		Boolean normalize = True
		String model = 'gamma_poisson'
		Int num_steps = 1500
		Float initial_lr = 1.0
		Float lr_decay_gamma = 0.001
		Float init_alpha = 100.0

		Int preemptible = 0
		Int disk_space = 50
		Int num_cpu = 16
		String memory = "120G"
		String docker_version = "numbat_wdl:0.0.2"
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
			output_directory = output_directory

			preemptible = preemptible,
			disk_space = disk_space,
			docker_version = docker_version,
			num_cpu = num_cpu,
			memory = memory,
			zones = zones
	}

	output {
		String output_directory = output_directory
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

		Int preemptible
		Int disk_space
		Int num_cpu
		String memory
		String docker_version
		String zones
	}

	command {
		python <<CODE

		import pickle
		import scanpy as sc
		from rich import print
		from scBTF import SingleCellTensor

		adata = sc.read('~{adata_path}')
		print(adata.shape)

		sc_tensor = SingleCellTensor.from_anndata(
			adata, sample_label='~{sample_label}',
			celltype_label='~{celltype_label}',
			scale_to='~{scale_to}', normalize='~{normalize}',
			hgnc_approved_genes_only='~{hgnc_approved_genes_only}',
			filter_gene_count='~{filter_gene_count}'
		)

		sc_tensor.tensor = sc_tensor.tensor.round()
		with open('results/tensor.pkl', 'wb') as file:
			pickle.dump(sc_tensor, file)
		print(f"Saved tensor of shape : {sc_tensor.tensor.shape} ")

		CODE
	}

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

	meta {
		author: "Daniel Chafamo"
		email : "chafamodaniel@gamil.com"
	}

}

task factorize_rank {

	input {
		File tensor_path
		Int rank
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

	command {
		python <<CODE

		import pickle
		from rich import print
		from scBTF import SingleCellBTF

		with open('~{tensor_path}', 'rb') as file:
			sc_tensor = pickle.load(file)
		print(f"Loaded tensor of shape : {sc_tensor.tensor.shape} ")

		if '~{method}' == "BTF":
			factorization_set = SingleCellBTF.factorize(
				sc_tensor=sc_tensor,
				rank='~{rank}',
				model='~{model}',
				n_restarts='~{n_restarts}',
				num_steps='~{num_steps}',
				init_alpha='~{init_alpha}',
				initial_lr='~{initial_lr}',
				lr_decay_gamma='~{lr_decay_gamma}',
				plot_var_explained=False
			)
		elif '~{method}' == "HALS":
			factorization_set = SingleCellBTF.factorize_hals(
				sc_tensor=sc_tensor,
				rank='~{rank}',
				num_steps='~{num_steps}',
				n_restarts='~{n_restarts}',
				sparsity_coefficients=[0., 0., 10.],
				plot_var_explained=False
			)

		factorization_set.save("results/rank_~{rank}_factorization.pkl")

		reconstructed_all = FactorizationSet()
		reconstructed_all.sc_tensor = factorization_set.sc_tensor

		for selected_rank in factorization_set.get_ranks():
			# Use median of clustered gene factors to reconstruct a final factorization
			medians = factorization_set.reconstruct_factors_from_median_gene_factor(rank = selected_rank)
			reconstructed = SingleCellBTF.factorize(
				sc_tensor = factorization_set.sc_tensor,
				rank = selected_rank,
				n_restarts = 1,
				init_alpha = '~{init_alpha}',
				num_steps = '~{num_steps}',
				initial_lr='~{initial_lr}',
				lr_decay_gamma='~{lr_decay_gamma}',
				model = 'gamma_poisson_fixed',
				fixed_mode = 2,
				fixed_value = torch.from_numpy(medians.T).float(),
				plot_var_explained = False
			)

			gene_factor = reconstructed.get_factorization(rank = selected_rank, restart_index = 0).gene_factor['mean'].numpy()
			print((1 - np.isclose(medians.T, gene_factor, atol=2)).sum(), '/', medians.flatten().shape[0], ' mismatches in final gene factors')
			print(f'variance explained by reconstructed factorization = {reconstructed.variance_explained(rank=selected_rank, restart_index=0).item() :.3}')
			reconstructed_all.factorizations[selected_rank] = reconstructed.factorizations[selected_rank]
		reconstructed_all.save('results/rank_~{rank}_consensus_factorization.pkl')

		CODE
	}

	output {
		File factorizations = "results/rank_~{rank}_factorization.pkl"
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

	meta {
		author: "Daniel Chafamo"
		email : "chafamodaniel@gamil.com"
	}

}


task aggregate_factorization {

	input {
		Array[File] consensus_factorizations
		String output_directory

		Int preemptible
		Int disk_space
		Int num_cpu
		String memory
		String docker_version
		String zones
	}

	command {
		python <<CODE

		from scBTF import SingleCellBTF

		reconstructed_all = FactorizationSet()

		for consensus_path in '~{sep="," consensus_factorizations}'.split(','):
			reconstructed = FactorizationSet.load(consensus_path)
			rank = reconstructed.get_ranks()[0]
			reconstructed_all.factorizations[rank] = reconstructed.factorizations[rank]

		reconstructed_all.sc_tensor = reconstructed.sc_tensor

		reconstructed_all.save('results/consensus_factorization.pkl')

		CODE

		gsutil -m cp -r "results/" ~{output_directory}/
	}

	output {
		File all_consensus_factorizations = 'results/consensus_factorization.pkl'
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
	meta {
		author: "Daniel Chafamo"
		email : "chafamodaniel@gamil.com"
	}
}