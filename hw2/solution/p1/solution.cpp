#include <mpi.h>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <sstream> 
#include <string>

#define USE_HDF5

#ifdef USE_HDF5
#include <hdf5.h>
#endif

using namespace std; 

#define L 1.0

struct WaveEquation 
{
  int N;       //grid points per dimension 
  double h;    //grid spacing (dx = dy = dz = h) 
  double dt;   //timestep
  double t;    //current time
  double * u;  //solution vector
  double * u_old;
  double * u_new;

  int Ntot;
  int procs_per_dim;

  int size;
  int rank;

  int rank_plus  [3];
  int rank_minus [3];
 
  MPI_Comm cart_comm; 

  double origin[3];
  double aux;
  int coords[3];

  ~WaveEquation()
  {
    delete [] u;
    delete [] u_old;
    delete [] u_new;
    MPI_Comm_free(&cart_comm);
  }

  WaveEquation (int a_N, int a_procs_per_dim)
  {
    MPI_Comm_size(MPI_COMM_WORLD,&size);
  

    procs_per_dim = a_procs_per_dim;
    Ntot = a_N;

    h  = L / a_N;  
    N = a_N / procs_per_dim;

    dt = h / sqrt(3.0);

    int nums    [3] = {procs_per_dim,procs_per_dim,procs_per_dim};
    int periodic[3] = {true,true,true};
    //int periodic[3] = {false,false,false};

    MPI_Cart_create(MPI_COMM_WORLD, 3, nums, periodic, true, &cart_comm);
    MPI_Comm_rank(cart_comm,&rank); 

    MPI_Cart_shift(cart_comm, 0, 1, &rank_plus[0], &rank_minus[0]);
    MPI_Cart_shift(cart_comm, 1, 1, &rank_plus[1], &rank_minus[1]);
    MPI_Cart_shift(cart_comm, 2, 1, &rank_plus[2], &rank_minus[2]);

    MPI_Cart_coords(cart_comm,rank,3,&coords[0]);


    origin[0] = coords[0] * N * h;
    origin[1] = coords[1] * N * h;
    origin[2] = coords[2] * N * h;

    u = new double [(N+2)*(N+2)*(N+2)];

    for (int i0=0; i0<N; i0++)
    for (int i1=0; i1<N; i1++)
    for (int i2=0; i2<N; i2++)
    {
      double x0 = origin[0] + i0*h + 0.5*h;
      double x1 = origin[1] + i1*h + 0.5*h;
      double x2 = origin[2] + i2*h + 0.5*h; 

      u[(i0+1)*(N+2)*(N+2)+(i1+1)*(N+2)+(i2+1)] = Initial_Condition(x0,x1,x2); 
    }

    u_old = new double [(N+2)*(N+2)*(N+2)];
    u_new = new double [(N+2)*(N+2)*(N+2)];

    for (int i0=1; i0<=N; i0++)
    for (int i1=1; i1<=N; i1++)
    for (int i2=1; i2<=N; i2++)
    {
      int m   = i2   + i1   *(N+2)+ i0  *(N+2)*(N+2);
      u_new[m] = u[m];
      u_old[m] = u[m];     
    }

    aux = dt*dt/h/h ;
  }

  double Initial_Condition(double x0, double x1, double x2)
  {
    double r = (x0-0.5)*(x0-0.5) + (x1-0.5)*(x1-0.5) ;//+ (x2-0.5)*(x2-0.5);
    return exp( - r /0.1 );
  }

  void Print(int kt = 0)
  {
    #ifdef USE_HDF5
    
         string name = "Wave__" + to_string(kt);
         if (rank == 0)
         {
           FILE *xmf = 0;
           xmf = fopen((name+".xmf").c_str(), "w");       
     
           std::stringstream s_head;
     
           s_head << "<?xml version=\"1.0\" ?>\n";
           s_head << "<!DOCTYPE Xdmf SYSTEM \"Xdmf.dtd\" []>\n";
           s_head << "<Xdmf Version=\"2.0\">\n";
           s_head << " <Domain>\n";
     
           s_head << "     <Time Value=\""<<std::setprecision(10)<<std::setw(10)<<t<<"\"/>\n\n";
     
           fprintf(xmf, (s_head.str()).c_str());     
             
           std::stringstream s;
           s << "   <Grid GridType=\"Uniform\">\n";
           s << "     <Topology TopologyType=\"3DCoRectMesh\" Dimensions=\" " <<Ntot +1 << " " <<Ntot + 1 << " " <<Ntot + 1 << "\"/>\n\n";
           s << "       <Geometry GeometryType=\"ORIGIN_DXDYDZ\">\n";
           s << "          <DataItem Dimensions=\"3\" NumberType=\"Double\" Precision=\"8\" Format=\"XML\">\n";
           s << "                 "<< 0 << " " << 0  << " " << 0 << "\n";
           s << "          </DataItem>\n";      
           s << "          <DataItem Dimensions=\"3\" NumberType=\"Double\" Precision=\"8\" Format=\"XML\">\n";
           s << "                 "<<std::setprecision(10) <<std::setw(15)<< h << " " << std::setw(15)<<h << " " << std::setw(15)<<h << "\n";
           s << "          </DataItem>\n";      
           s << "     </Geometry>\n\n";
     
     
           s << "     <Attribute Name=\"data\" AttributeType=\" "<< "Scalar" << "\" Center=\"Cell\">\n";
           s << "       <DataItem Dimensions=\" " <<Ntot << " " << Ntot << " " << Ntot << " " << std::setw(10)<<1 <<"\" NumberType=\"Float\" Precision=\" " <<   sizeof(double)  << "\" Format=\"HDF\">\n";           
           std::stringstream name_ss;
           name_ss <<"data" ; // << std::setfill('0') << std::setw(10) << mm ;
           std::string tmp = name_ss.str();
           s << "        "<<(name+".h5").c_str()<<":/"<<tmp<<"\n";        
           s << "       </DataItem>\n";
           s << "     </Attribute>\n";
     
           s << "   </Grid>\n";
           
           std::string st = s.str();
           
           fprintf(xmf, st.c_str());         
     
     
           std::stringstream s_tail;
           s_tail <<  " </Domain>\n";
           s_tail <<  "</Xdmf>\n";
               
           fprintf(xmf, (s_tail.str()).c_str());
           fclose(xmf);
         }
     
         hid_t file_id, dataset_id, fspace_id, fapl_id, mspace_id;
     
         H5open();
         fapl_id = H5Pcreate(H5P_FILE_ACCESS);
     
         H5Pset_fapl_mpio(fapl_id, cart_comm, MPI_INFO_NULL); 
             
         file_id = H5Fcreate((name + ".h5").c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, fapl_id);
         
         H5Pclose(fapl_id);
         
         double * array_all = new double[N * N * N];
     
         hsize_t count [4] = {static_cast<unsigned int>(N)            , static_cast<unsigned int>(N)            , static_cast<unsigned int>(N)            , 1};
         hsize_t dims  [4] = {static_cast<unsigned int>(Ntot)         , static_cast<unsigned int>(Ntot)         , static_cast<unsigned int>(Ntot)         , 1};
         hsize_t offset[4] = {static_cast<unsigned int>(coords[0] * N), static_cast<unsigned int>(coords[1] * N), static_cast<unsigned int>(coords[2] * N), 0};
     
         for(int i0=1; i0<=N; i0++)
         {
           for(int i1=1; i1<=N; i1++)
           {
             for(int i2=1; i2<=N; i2++)
             {
               array_all[(i2-1)+(i1-1)*N + (i0-1)*N*N] = u[i2 + i1*(N+2)+i0*(N+2)*(N+2)];
             }
           }
         }
     
         fapl_id = H5Pcreate(H5P_DATASET_XFER);   
         H5Pset_dxpl_mpio(fapl_id, H5FD_MPIO_COLLECTIVE);
         fspace_id = H5Screate_simple(4, dims, NULL);
     
         dataset_id = H5Dcreate(file_id, "data", H5T_NATIVE_DOUBLE, fspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
     
         fspace_id = H5Dget_space(dataset_id);
         
         H5Sselect_hyperslab(fspace_id, H5S_SELECT_SET, offset, NULL, count, NULL);
         
         mspace_id = H5Screate_simple(4, count, NULL);
         
     
         H5Dwrite(dataset_id, H5T_NATIVE_DOUBLE, mspace_id, fspace_id, fapl_id, array_all);
     
         H5Sclose(mspace_id );
         H5Sclose(fspace_id );
         H5Dclose(dataset_id);
         H5Pclose(fapl_id   );
         H5Fclose(file_id   );
         H5close();
     
         delete [] array_all;

    #else

         string a_name = "Wave__" + to_string(kt); 
         char name [30] = a_name.c_str(); 
         MPI_File file;
         MPI_File_open( cart_comm, name, MPI_MODE_RDWR | MPI_MODE_CREATE , MPI_INFO_NULL, &file);
     
         stringstream ss;
     
         
         for (int i0=0; i0<N; i0++)
         {
           double x0 = origin[0] + i0*h + 0.5*h;
           for (int i1=0 ; i1<N; i1++)
           {
             double x1 = origin[1] + i1*h + 0.5*h;
             for (int i2=0; i2<N; i2++)
             {
               double x2 = origin[2] + i2*h + 0.5*h;
               ss << x0 << " " << x1 << " " << x2 << " " << u[(i2+1) + (i1+1)*(N+2) + (i0+1)*(N+2)*(N+2)] << "\n";
             }
           }
         }
         
         std::string asciiData = ss.str();
         
         MPI_Offset my_length = asciiData.size()*sizeof(char);
     
         MPI_Offset offset;
         MPI_Exscan(&my_length, &offset, 1, MPI_OFFSET, MPI_SUM, cart_comm);
     
         MPI_File_write_at_all(file, offset, asciiData.data(), asciiData.size(), MPI_CHAR, MPI_STATUS_IGNORE);
         MPI_File_close(&file);
     

    #endif
  }

  void UpdateGridPoint(int i0, int i1, int i2)
  {
        int m   = i0   *(N+2)*(N+2)+ i1   *(N+2)+ i2   ;
        int ip1 =(i0+1)*(N+2)*(N+2)+(i1  )*(N+2)+(i2  );
        int im1 =(i0-1)*(N+2)*(N+2)+(i1  )*(N+2)+(i2  );
        int jp1 =(i0  )*(N+2)*(N+2)+(i1+1)*(N+2)+(i2  );
        int jm1 =(i0  )*(N+2)*(N+2)+(i1-1)*(N+2)+(i2  );
        int kp1 =(i0  )*(N+2)*(N+2)+(i1  )*(N+2)+(i2+1);
        int km1 =(i0  )*(N+2)*(N+2)+(i1  )*(N+2)+(i2-1);
        u_new[m] = 2.0*u[m] - u_old[m] + aux*(u[ip1]+u[im1]+u[jp1]+u[jm1]+u[kp1]+u[km1]-6.0*u[m]);   
  }


  void run(double t_end)
  {   
    t = 0;

    MPI_Datatype SEND_FACE_PLUS [3];
    MPI_Datatype SEND_FACE_MINUS[3]; 

    MPI_Datatype RECV_FACE_PLUS [3];
    MPI_Datatype RECV_FACE_MINUS[3]; 
    

    int ndims = 3;
    int order =  MPI_ORDER_C;
    int array_of_sizes [3] = {N+2,N+2,N+2};
    int array_of_subsizes [3];
    int array_of_starts [3];  
    for (int d0 = 0; d0 < 3; d0++)
    {
    	int d1 = (d0+1)%3;
    	int d2 = (d0+2)%3;

    	array_of_subsizes[d0] = 1;
    	array_of_subsizes[d1] = N;
    	array_of_subsizes[d2] = N;

    	array_of_starts[d0] = 1;
    	array_of_starts[d1] = 1;
    	array_of_starts[d2] = 1;
    	MPI_Type_create_subarray(ndims,array_of_sizes,array_of_subsizes,array_of_starts,order,MPI_DOUBLE,&SEND_FACE_MINUS[d0]);
    
    	array_of_starts[d0] = N;
    	array_of_starts[d1] = 1;
    	array_of_starts[d2] = 1;
    	MPI_Type_create_subarray(ndims,array_of_sizes,array_of_subsizes,array_of_starts,order,MPI_DOUBLE,&SEND_FACE_PLUS[d0]);


    	array_of_starts[d0] = 0;
    	array_of_starts[d1] = 1;
    	array_of_starts[d2] = 1;
      MPI_Type_create_subarray(ndims,array_of_sizes,array_of_subsizes,array_of_starts,order,MPI_DOUBLE,&RECV_FACE_MINUS[d0]);

    	array_of_starts[d0] = N+1;
    	array_of_starts[d1] = 1;
    	array_of_starts[d2] = 1;
      MPI_Type_create_subarray(ndims,array_of_sizes,array_of_subsizes,array_of_starts,order,MPI_DOUBLE,&RECV_FACE_PLUS[d0]);

      MPI_Type_commit(&SEND_FACE_PLUS [d0]); MPI_Type_commit(&RECV_FACE_PLUS [d0]);
      MPI_Type_commit(&SEND_FACE_MINUS[d0]); MPI_Type_commit(&RECV_FACE_MINUS[d0]);
    }


    int count = 0;
    do
    {    

      if (count % 100 == 0)
      {
        if (rank==0)cout << count << " t=" << t << "\n";
        Print(count);
      } 

      MPI_Request request[12];
      for (int d=0; d<3; d++)
      {
        MPI_Irecv(u,1,RECV_FACE_MINUS[d],rank_minus[d],100*d  ,cart_comm,&request[2*d  ]);     
        MPI_Irecv(u,1,RECV_FACE_PLUS [d],rank_plus [d],100*d+1,cart_comm,&request[2*d+1]);
      }

      for (int d=0; d<3; d++)
      {
        MPI_Isend(u,1,SEND_FACE_PLUS [d],rank_plus [d],100*d  ,cart_comm,&request[6+2*d  ]);
        MPI_Isend(u,1,SEND_FACE_MINUS[d],rank_minus[d],100*d+1,cart_comm,&request[6+2*d+1]);
      }

      for (int i0=1 + 1; i0<=N - 1; i0++)
      for (int i1=1 + 1; i1<=N - 1; i1++)
      for (int i2=1 + 1; i2<=N - 1; i2++)
        UpdateGridPoint(i0,i1,i2);

      MPI_Waitall(12,&request[0],MPI_STATUSES_IGNORE);     

      
      int i,j,k;
 
      k = 1;
      for (int j=1; j<=N; j++)
      for (int i=1; i<=N; i++)
        UpdateGridPoint(i,j,k); 
      
      k = N;
      for (int j=1; j<=N; j++)
      for (int i=1; i<=N; i++)
        UpdateGridPoint(i,j,k); 
      
      j = 1;
      for (int k=1+1; k<=N-1; k++)
      for (int i=1  ; i<=N  ; i++)
        UpdateGridPoint(i,j,k); 
      
      j = N;
      for (int k=1+1; k<=N-1; k++)
      for (int i=1  ; i<=N  ; i++)
        UpdateGridPoint(i,j,k); 
      
      i = 1;
      for (int k=1+1; k<=N-1; k++)
      for (int j=1+1; j<=N-1; j++)
        UpdateGridPoint(i,j,k); 
      
      i = N;
      for (int k=1+1; k<=N-1; k++)
      for (int j=1+1; j<=N-1; j++)
        UpdateGridPoint(i,j,k); 
      

      double * temp = u_old;
      u_old = u;
      u     = u_new;
      u_new = temp; 

      t += dt;
      count ++;

    }while(t < t_end);

    double s =0;
    double Checksum =0;
    for (int k=1; k<=N; k++)
    for (int j=1; j<=N; j++)
    for (int i=1; i<=N; i++)
    {
      int m   = k   + j   *(N+2)+ i   *(N+2)*(N+2);
      s +=  u[m]*u[m];     
    }

    MPI_Reduce(&s,&Checksum,1,MPI_DOUBLE,MPI_SUM,0,cart_comm);
    if (rank == 0)
    std::cout << "Checksum = " << Checksum<< "\n";

    for (int d=0; d<3; d++)
    {
        MPI_Type_free(&RECV_FACE_PLUS[d]) ;
        MPI_Type_free(&RECV_FACE_MINUS[d]); 	
        MPI_Type_free(&SEND_FACE_PLUS[d]) ;
        MPI_Type_free(&SEND_FACE_MINUS[d]);
    }
  }
};


int main(int argc, char ** argv)
{
	MPI_Init(&argc, &argv);

    int rank,size;
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    MPI_Comm_size(MPI_COMM_WORLD,&size);

	if (argc != 4)
	{
        if (rank==0)
        {
            std::cout << " Incorrect number of inputs. Run as:\n";
            std::cout << " mpirun -n x ./Wave2D N y t \n";
            std::cout << " where \n";
            std::cout << " N= number of grid points per direction\n";
            std::cout << " y= number of ranks per direction (=sqrt(x))\n";
            std::cout << " t= final time of simulation \n";
            std::cout << " Aborting...\n";
        }
        int err = 1;
        MPI_Abort(MPI_COMM_WORLD,err);	
	}

	int points        =  atoi(argv[1]);
	int procs_per_dim =  atoi(argv[2]);
	double t_end      =  atof(argv[3]);


	if (size != procs_per_dim*procs_per_dim*procs_per_dim)
    {
        if (rank==0)
            std::cout << " Incorrect number of ranks per direction. Aborting... \n";
        int err = 2;
        MPI_Abort(MPI_COMM_WORLD,err);
    }

    if (points % procs_per_dim != 0)
    {
        if (rank==0)
    	    std::cout << " Grid points per direction must be divisible by the number of ranks per direction. Aborting... \n";
    	int err = 3;
    	MPI_Abort(MPI_COMM_WORLD,err);      
    }

    WaveEquation * Simulation = new WaveEquation (points,procs_per_dim);

    
    double start = MPI_Wtime();
    Simulation->run(t_end);
    double finish = MPI_Wtime();

    if (rank==0)std::cout << "Total time = " << finish - start << "\n";

    delete Simulation;

    MPI_Finalize();
    return 0;
}