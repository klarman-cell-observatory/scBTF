Run scBTF on Terra
---------------------------------------

Pegasus can be used as a command line tool. Type::

	pegasus -h

to see the help information::

	Usage:
		pegasus <command> [<args>...]
		pegasus -h | --help
		pegasus -v | --version

``pegasus`` has 9 sub-commands in 6 groups.

* Preprocessing:

	aggregate_matrix
		Aggregate sample count matrices into a single count matrix. It also enables users to import metadata into the count matrix.

* Demultiplexing:

	demuxEM
		Demultiplex cells/nuclei based on DNA barcodes for cell-hashing and nuclei-hashing data.

* Analyzing:

	cluster
		Perform first-pass analysis using the count matrix generated from 'aggregate_matrix'. This subcommand could perform low quality cell filtration, batch correction, variable gene selection, dimension reduction, diffusion map calculation, graph-based clustering, visualization. The final results will be written into zarr-formatted file, or h5ad file, which Seurat could load.

	de_analysis
		Detect markers for each cluster by performing differential expression analysis per cluster (within cluster vs. outside cluster). DE tests include Welch's t-test, Fisher's exact test, Mann-Whitney U test. It can also calculate AUROC values for each gene.

	find_markers
		Find markers for each cluster by training classifiers using LightGBM.

	annotate_cluster
		This subcommand is used to automatically annotate cell types for each cluster based on existing markers. Currently, it works for human/mouse immune/brain cells, etc.

* Plotting:

	plot
		Make static plots, which includes plotting tSNE, UMAP, and FLE embeddings by cluster labels and different groups.

* Web-based visualization:

	scp_output
		Generate output files for single cell portal.

* MISC:

	check_indexes
		Check CITE-Seq/hashing indexes to avoid index collision.

---------------------------------


Quick guide
^^^^^^^^^^^

Suppose you have ``example.csv`` ready with the following contents::

	Sample,Source,Platform,Donor,Reference,Location
	sample_1,bone_marrow,NextSeq,1,GRCh38,/my_dir/sample_1/raw_feature_bc_matrices.h5
	sample_2,bone_marrow,NextSeq,2,GRCh38,/my_dir/sample_2/raw_feature_bc_matrices.h5
	sample_3,pbmc,NextSeq,1,GRCh38,/my_dir/sample_3/raw_gene_bc_matrices_h5.h5
	sample_4,pbmc,NextSeq,2,GRCh38,/my_dir/sample_4/raw_gene_bc_matrices_h5.h5

You want to analyze all four samples but correct batch effects for bone marrow and pbmc samples separately. You can run the following commands::

	pegasus aggregate_matrix --attributes Source,Platform,Donor example.csv example.aggr
	pegasus cluster -p 20 --correct-batch-effect --batch-group-by Source --louvain --umap example.aggr.zarr.zip example
	pegasus de_analysis -p 20 --labels louvain_labels example.zarr.zip example.de.xlsx
	pegasus annotate_cluster example.zarr.zip example.anno.txt
	pegasus plot compo --groupby louvain_labels --condition Donor example.zarr.zip example.composition.pdf
	pegasus plot scatter --basis umap --attributes louvain_labels,Donor example.zarr.zip example.umap.pdf

The above analysis will give you UMAP embedding and Louvain cluster labels in ``example.zarr.zip``, along with differential expression analysis
result stored in ``example.de.xlsx``, and putative cluster-specific cell type annotation stored in ``example.anno.txt``.
You can investigate donor-specific effects by looking at ``example.composition.pdf``.
``example.umap.pdf`` plotted UMAP colored by louvain_labels and Donor info side-by-side.
