/****************************************************************/
/* Parallel Combinatorial BLAS Library (for Graph Computations) */
/* version 1.6 -------------------------------------------------*/
/* date: 6/15/2017 ---------------------------------------------*/
/* authors: Ariful Azad, Aydin Buluc  --------------------------*/
/****************************************************************/
/*
 Copyright (c) 2010-2017, The Regents of the University of California
 
 Permission is hereby granted, free of charge, to any person obtaining a copy
 of this software and associated documentation files (the "Software"), to deal
 in the Software without restriction, including without limitation the rights
 to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 copies of the Software, and to permit persons to whom the Software is
 furnished to do so, subject to the following conditions:
 
 The above copyright notice and this permission notice shall be included in
 all copies or substantial portions of the Software.
 
 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 THE SOFTWARE.
 */

#define DETERMINISTIC
#include "CombBLAS/CombBLAS.h"
#include <mpi.h>
#include <sys/time.h> 
#include <iostream>
#include <functional>
#include <algorithm>
#include <vector>
#include <string>
#include <sstream>
#include <ranges>

#include <kagen.h>

double cblas_alltoalltime;
double cblas_allgathertime;
double cblas_mergeconttime;
double cblas_transvectime;
double cblas_localspmvtime;

#define ITERS 1
using namespace std;
using namespace combblas;

int main(int argc, char* argv[])
{
	int nprocs, myrank;
	MPI_Init(&argc, &argv);
    
	MPI_Comm_size(MPI_COMM_WORLD,&nprocs);
	MPI_Comm_rank(MPI_COMM_WORLD,&myrank);
	if(argc < 3)
	{
		if(myrank == 0)
		{
			cout << "Usage: ./ccbfs <KaGen> <KaGen Options>" << endl;
			cout << "Example: ./ccbfs KaGen 'gnm-undirected;n=12;m=15'" << endl;
		}
		MPI_Finalize();
		return -1;
	}		
	{
		typedef SelectMaxSRing<bool, int32_t> SR;
		typedef SpParMat < int64_t, bool, SpDCCols<int64_t,bool> > PSpMat_Bool;
		typedef SpParMat < int64_t, bool, SpDCCols<int32_t,bool> > PSpMat_s32p64;	// sequentially use 32-bits for local matrices, but parallel semantics are 64-bits
		typedef SpParMat < int64_t, int, SpDCCols<int32_t,int> > PSpMat_s32p64_Int;	// similarly mixed, but holds integers as upposed to booleans
		typedef SpParMat < int64_t, int64_t, SpDCCols<int64_t,int64_t> > PSpMat_Int64;
		shared_ptr<CommGrid> fullWorld;
		fullWorld.reset( new CommGrid(MPI_COMM_WORLD, 0, 0) );

		// Declare objects
		PSpMat_Bool A(fullWorld);
		PSpMat_s32p64 Aeff(fullWorld);
		FullyDistVec<int64_t, int64_t> degrees(fullWorld);	// degrees of vertices (including multi-edges and self-loops)
		OptBuf<int32_t, int64_t> optbuf;	// let indices be 32-bits

		if(string(argv[1]) == string("KaGen"))
		{
			string options = string(argv[2]);
			
			// Initialize KaGen
			kagen::KaGen gen(MPI_COMM_WORLD);
			gen.SetSeed(1); // Use deterministic seed for reproducibility
			gen.EnableUndirectedGraphVerification();
			gen.UseEdgeListRepresentation();

			// Generate graph using option string
			kagen::Graph graph = gen.GenerateFromOptionString(options);

			// Convert KaGen graph to CombBLAS format using clean API
			auto globaln = graph.NumberOfGlobalVertices();
			auto edges = graph.TakeEdges();

			// Transform edges to CombBLAS format using ranges
			auto edge_list = edges | 
				std::views::transform([](const auto& edge) {
					return std::tuple<int64_t, int64_t, bool>(edge.first, edge.second, true);
				});
			
			// Create the CombBLAS matrix from the local edge list
			A.ReadFromLocalEdgeList(edge_list, globaln, combblas::maximum<bool>());
			
			// Calculate degrees before conversion
			PSpMat_Int64 * G = new PSpMat_Int64(A); 
			G->Reduce(degrees, Row, plus<int64_t>(), static_cast<int64_t>(0));	// identity is 0 
			delete G;

			// Convert to efficient format
			Aeff = PSpMat_s32p64(A);
			A.FreeMemory();
			
			Aeff.OptimizeForGraph500(optbuf);
		}
		else 
		{
			if(myrank == 0)
			{
				cout << "Unknown option. Only KaGen is supported." << endl;
			}
			MPI_Finalize();
			return -1;	
		}

		// Set to true to output graph verification information
		bool verify_graph = false;
		
		// Output graph structure for verification if requested
		if (verify_graph) {
			int64_t total_vertices = degrees.TotalLength();
			int64_t total_edges = Aeff.getnnz();
			
			if(myrank == 0) {
				cout << "\n=== Graph Verification Output ===" << endl;
				cout << "Total vertices: " << total_vertices << endl;
				cout << "Total edges: " << total_edges << endl;
			}
			
			// Each rank outputs its local matrix information
			for(int r = 0; r < nprocs; ++r) {
				if(myrank == r) {
					// Get local matrix dimensions
					int64_t local_rows = Aeff.getlocalrows();
					int64_t local_cols = Aeff.getlocalcols();
					int64_t local_nnz = Aeff.getlocalnnz();
					
					cout << "Rank " << r << " local matrix: " 
					     << local_rows << " rows x " << local_cols << " cols, "
					     << local_nnz << " non-zeros" << endl;
					
					// Output some local degree information
					int64_t first_vertex_idx = degrees.LocArrSize() > 0 ? degrees.LengthUntil() : 0;
					int64_t num_local_vertices = degrees.LocArrSize();
					
					if(num_local_vertices > 0) {
						cout << "  Local vertex range: [" << first_vertex_idx 
						     << ", " << (first_vertex_idx + num_local_vertices) << ")" << endl;
						
						// Print first few vertices and their degrees
						int64_t max_vertices_to_print = min((int64_t)10, num_local_vertices);
						cout << "  First " << max_vertices_to_print << " vertices and degrees:" << endl;
						for(int64_t i = 0; i < max_vertices_to_print; ++i) {
							cout << "    Vertex " << (first_vertex_idx + i) 
							     << " degree: " << degrees.GetLocalElement(i) << endl;
						}
					}
				}
				MPI_Barrier(MPI_COMM_WORLD);
			}
			
			if(myrank == 0) {
				cout << "=== End Verification Output ===" << endl << endl;
			}
		}

		MPI_Barrier(MPI_COMM_WORLD);
		double t1 = MPI_Wtime();

		double nver = (double) degrees.TotalLength();

		MPI_Pcontrol(1,"BFSCC");
		
		FullyDistVec<int64_t, int64_t> parents ( Aeff.getcommgrid(), Aeff.getncol(), (int64_t) -1);	// identity is -1
		uint64_t num_components = 0;
		double cc_start = MPI_Wtime();
		
		for(int64_t vertex = 0; vertex < static_cast<int64_t>(nver);) 
		{
			cblas_allgathertime = 0;
			cblas_alltoalltime = 0;
			cblas_mergeconttime = 0;
			cblas_transvectime  = 0;
			cblas_localspmvtime = 0;
			MPI_Barrier(MPI_COMM_WORLD);
			double bfs_iteration_start = MPI_Wtime();
			++num_components;

			FullyDistSpVec<int64_t, int64_t> fringe(Aeff.getcommgrid(), Aeff.getncol());	// numerical values are stored 0-based
			fringe.SetElement(vertex, vertex);
			
			int step = 0;
			while(fringe.getnnz() > 0)
			{
				fringe.setNumToInd();
				fringe = SpMV(Aeff, fringe, optbuf);	// SpMV with sparse vector (with indexisvalue flag preset), optimization enabled
				fringe = EWiseMult(fringe, parents, true, (int64_t) -1);	// clean-up vertices that already has parents 
				parents.Set(fringe);
				step++;
			}
			MPI_Barrier(MPI_COMM_WORLD);
			double bfs_iteration_end = MPI_Wtime();
			
			// Find next unvisited vertex
			auto [next_vertex, parent] = parents.MinElement();
			while (next_vertex == vertex) {
				parents.SetElement(vertex, vertex);
				std::tie(next_vertex, parent) = parents.MinElement();
				if (parent != -1) {
					break;
				}
			}
			if (parent != -1) {
				break;
			}
			vertex = next_vertex;
		}
		
		double cc_end = MPI_Wtime();
		SpParHelper::Print("Finished\n");
		SpParHelper::Print("Number of components: " + to_string(num_components) + "\n");
		ostringstream os;
		MPI_Pcontrol(-1,"BFSCC");

		os << "CC runtime: " << (cc_end - cc_start) << "\n";
		SpParHelper::Print(os.str());
	}
	MPI_Finalize();
	return 0;
}


