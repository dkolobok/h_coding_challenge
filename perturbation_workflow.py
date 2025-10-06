"""
Perturbation Workflow Module

This module provides a workflow for perturbing gene expression in single-cell RNA-seq data
and optionally generating embeddings using foundation models.

Classes:
    PerturbationWorkflow: Main class for performing gene perturbations on AnnData objects
"""

import logging
from scipy import sparse
import numpy as np

class PerturbationWorkflow:
    """
    A workflow class for perturbing gene expression in single-cell RNA-seq data.
    
    This class allows in-silico perturbation of gene expression levels by applying
    fold changes to specific genes in an AnnData object. It can optionally generate
    embeddings using foundation models like Geneformer after perturbation.
    
    Attributes:
        adata (anndata.AnnData): The single-cell RNA-seq dataset to perturb
    """
    
    def __init__(self, adata):
        """
        Initialize the PerturbationWorkflow with a single-cell dataset.
        
        Args:
            adata (anndata.AnnData): The input AnnData object containing gene expression data
        """
        self.adata = adata

    def perturb_single_gene(self, gene_name, fold_change, helical_model=None, save_path=None):
        """
        Perturb a single gene by applying a fold change to its expression values.
        
        This method multiplies the expression values of a specified gene by a fold change
        factor across all cells. Optionally, it can generate embeddings using a foundation
        model and save the results.
        
        Args:
            gene_name (str): Name of the gene to perturb (must exist in adata.var_names)
            fold_change (float): Multiplicative factor to apply to gene expression.
                                 - fold_change > 1: overexpression
                                 - fold_change < 1: downregulation
                                 - fold_change = 0: knockout
            helical_model (optional): A helical foundation model (e.g., Geneformer) for 
                                      generating embeddings. If None, returns perturbed AnnData
            save_path (str, optional): Path to save the output. For embeddings, saves as .npz;
                                       for AnnData, saves as .h5ad
        
        Returns:
            numpy.ndarray: Embeddings array if helical_model is provided
            anndata.AnnData: Perturbed AnnData object if helical_model is None
        
        Raises:
            ValueError: If the specified gene is not found in the dataset
            AssertionError: If the data matrix is not in sparse format
        
        Note:
            - Currently only supports sparse matrices
            - Perturbation info is stored in adata.uns['perturbation']
        
        TODO:
            - Add support for perturbing using a custom function rather than just fold change
            - Add support for perturbing a subset of cells
        """
        logging.info(f"Perturbing gene {gene_name} with fold change {fold_change}")
        
        # Create a copy to avoid modifying the original data
        perturbed_adata = self.adata.copy()
        
        # Validate that the gene exists in the dataset
        if gene_name not in perturbed_adata.var_names:
            raise ValueError(f"Gene {gene_name} not found in the dataset.")

        # Get the index of the gene to perturb
        gene_idx = perturbed_adata.var_names.get_loc(gene_name)
        
        # Apply perturbation to the gene expression matrix
        assert sparse.issparse(perturbed_adata.X), "This function currently only supports sparse matrices."
        
        # Convert to CSC (Compressed Sparse Column) format for efficient column operations
        logging.info(f"Converting to csc format...")
        perturbed_adata.X = perturbed_adata.X.tocsc(copy=True)
        
        # Apply fold change to the target gene across all cells
        logging.info(f"Multiplying by fold change...")
        perturbed_adata.X[:, gene_idx] = perturbed_adata.X[:, gene_idx].multiply(fold_change)
        
        # Convert back to CSR (Compressed Sparse Row) format for efficient row operations
        logging.info(f"Converting back to csr format...")
        perturbed_adata.X = perturbed_adata.X.tocsr()

        # Store perturbation metadata for tracking
        perturbed_adata.uns['perturbation'] = {
            'gene': gene_name,
            'fold_change': fold_change
        }

        # If a foundation model is provided, generate embeddings
        if helical_model is not None:
            logging.info(f"Generating embeddings using Helical model...")
            embeddings = helical_model.get_embeddings(helical_model.process_data(perturbed_adata))
            logging.info(f"Embeddings generated with shape: {embeddings.shape}")
            
            # Save embeddings if path is provided
            if save_path is not None:
                logging.info(f"Saving perturbed embeddings to {save_path}...")
                np.savez_compressed(save_path, embeddings=embeddings)
            
            # Return embeddings directly to save memory (avoid storing full gene expression data)
            return embeddings
        
        # Save perturbed AnnData if path is provided and no model is used
        if save_path is not None:
            logging.info(f"Saving perturbed AnnData to {save_path}...")
            perturbed_adata.write_h5ad(save_path)
        
        return perturbed_adata
    
    def perturb_batch(self, perturbation_list):
        """
        Perform multiple gene perturbations in batch.
        
        This method allows running multiple perturbations sequentially using a list of
        perturbation specifications. Each perturbation is independent and creates a fresh
        copy from the original dataset.
        
        Args:
            perturbation_list (list): List of dictionaries, where each dictionary contains:
                - 'gene_name' (str): Name of the gene to perturb
                - 'fold_change' (float): Fold change to apply
                - 'helical_model' (optional): Foundation model for embeddings
                - 'save_path' (str, optional): Path to save output
        
        Returns:
            list: List of results (either embeddings or AnnData objects) for each perturbation
        
        Example:
            >>> perturbations = [
            ...     {"gene_name": "SOD1", "fold_change": 0.5, "helical_model": model, 
            ...      "save_path": "output/SOD1_perturbed.npz"},
            ...     {"gene_name": "FUS", "fold_change": 2.0, "helical_model": model,
            ...      "save_path": "output/FUS_perturbed.npz"}
            ... ]
            >>> results = workflow.perturb_batch(perturbations)
        """
        results = []
        for perturbation in perturbation_list:
            gene_name = perturbation['gene_name']
            fold_change = perturbation['fold_change']
            helical_model = perturbation.get('helical_model', None)
            save_path = perturbation.get('save_path', None)

            perturbation_result = self.perturb_single_gene(
                gene_name=gene_name, 
                fold_change=fold_change, 
                helical_model=helical_model, 
                save_path=save_path
            )
            results.append(perturbation_result)

        return results
    