#include <cmath>
#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <cstdlib>
#include <cstdio>
#include <random>
#include <algorithm>
#include <ctime>

#define __GPU__
#include "../include/Lattice_geometry.h"
#include "../include/Eigen_system.h"
#include "../include/Equations.h"
#include "../include/Optimizer.h"
#include "../include/argparse.h"

#define __OUTFILE__

using LATTICE = KAGOMEv2;
using namespace BdG_Tech_RealPair;

std::vector<std::vector<int>> make_lattice(int l);

struct PROFILE
{
    const static int l=12;
    const static int L=3*l*l;
    const static int L2=L*2;
    const static int N_tot = 720; // # of Total electrons
    constexpr static double RBZ_x=2./(double)l, RBZ_y=2./(double)l; // For unit cell : RBZ_x=2, RBZ_y=2
    constexpr static double V=1.7320508076*2*l*l;
    static std::vector<std::vector<int>> lattices;
};

std::vector<std::vector<int>> PROFILE::lattices = make_lattice(PROFILE::l);

using EIGEN = EigenSetMCM<LATTICE,PROFILE>;
using EQUATION = Gauss<EIGEN,PROFILE>;

std::string remove_zeros_in_str(const double val);

int main(int argc, char* argv[])
{
    std::vector<pair_t> options, defaults;
    // env; explanation of env
    options.push_back(pair_t("U", "interaction strength"));
    options.push_back(pair_t("X", "disorder strength"));
    options.push_back(pair_t("Nsam", "# of samples"));
    options.push_back(pair_t("Ndev", "# of devices"));
    options.push_back(pair_t("path", "directory to load and save files"));
    options.push_back(pair_t("xpath", "directory to load parameter's files"));
    // env; default value
    defaults.push_back(pair_t("X", "0"));
    defaults.push_back(pair_t("Ndev", "8"));
    defaults.push_back(pair_t("Nsam", "300"));
    defaults.push_back(pair_t("path", "."));
    // parser for arg list
    argsparse parser(argc, argv, options, defaults);

    const int L = PROFILE::L,
        L2 = PROFILE::L2,
        l = PROFILE::l,
        N_tot = PROFILE::N_tot;
    const int X = parser.find<int>("X"),
        Nsam = parser.find<int>("Nsam"),
        Ndev = parser.find<int>("Ndev");
    const double U = parser.find<double>("U"),
        beta = 200,
        V = PROFILE::V;

    int ranks, nprocs;

    MPIERR(MPI_Init(&argc,&argv)); 
    MPIERR(MPI_Comm_size(MPI_COMM_WORLD, &nprocs));
    MPIERR(MPI_Comm_rank(MPI_COMM_WORLD, &ranks));

    if (Ndev >= nprocs) {
        cudaSetDevice(ranks);
    } else {
        std::cerr << "nprocs is over # of devices!" << std::endl;
        exit(1);
    }

    if (ranks==0) parser.print(std::cout);

    double gamma[L];

    EIGEN eigen(L,gamma);
    EQUATION equation(eigen,L,45);

    std::vector<double> m(L), dilUsite(L,U), d(L);
        
    std::string filename = parser.find<>("path") + "/RandomHopKagome_SFW_v2-L" + std::to_string(L) + "U" + parser.find<>("U") + "X" + parser.find<>("X") + "N" + std::to_string(N_tot) + ".dat"; 
#ifdef __OUTFILE__
    std::ofstream outfile;
    bool outfileflag;
    if (ranks==0) 
        outfileflag = std::ifstream(filename).is_open();
    MPIERR(MPI_Bcast(&outfileflag,1,MPI_CXX_BOOL,0,MPI_COMM_WORLD));
    if (!outfileflag)
    {
        outfile.open(filename);
        if (ranks==0) 
            outfile << "#        D^s_conv        D^s_geom         D^s       " << std::endl;
    }
    else
        outfile.open(filename); 
    outfile.precision(10);
#else 
    if (ranks==0)
        std::cout << "#        D^s_conv        D^s_geom         D^s       " << std::endl;
#endif
    for (int i=0; i<Nsam; ++i)
    {
        std::ifstream lfile;
        std::string lfilename = parser.find<>("xpath") + "/KagomeProfile-L" + std::to_string(L) + "U" + parser.find<>("U") + "X" + parser.find<>("X") + "N" + std::to_string(N_tot) + "-TAG" + std::to_string(i) + ".dat";

        if (!std::ifstream(lfilename).is_open())
        {
            std::cerr << "There is not the file!" << std::endl;
            exit(1);
        }
        else
        {
            lfile.open(lfilename);

            std::string Line;
            double LineParam[7];
            for (int j=0;j<L;++j)
            {
                std::getline(lfile,Line);
                std::istringstream iss(Line);

                if (Line[0] != '#')
                {
                    for (int jj=0;jj<7;++jj) iss >> LineParam[jj];
                    gamma[j] = LineParam[2];
                    d[j] = LineParam[3]*dilUsite[j];
                    m[j] = LineParam[5];
                }
                else
                {
                    --j;
                    continue;
                }
            }
            lfile.close();
        }

        std::pair<double,double> sfw = equation.SFW(m,d,V,beta,ranks,nprocs);
#ifdef __OUTFILE__		
        if(ranks == 0)
        {
            outfile << "\t" << sfw.first << "\t" << sfw.second << "\t" << sfw.first+sfw.second << std::endl;
            std::cout << "\r# --- Estimating ... " << std::setw(7) << i+1 << std::flush; 
        }
#else
        if(ranks == 0)
        {
            std::cout << "\t" << sfw.first << "\t" << sfw.second << "\t" << sfw.first+sfw.second << std::endl;
        }
#endif
    }
#ifdef __OUTFILE__
    outfile.close();
#endif
    MPIERR(MPI_Finalize());

    return 0;
}

std::string remove_zeros_in_str(const double val)
{
    std::string tmp = std::to_string(val);
    int zero_pos = tmp.find_last_not_of('0'), point_pos = tmp.find_last_not_of('.');
    if (zero_pos-point_pos == 1){
        tmp.erase(zero_pos + 2, std::string::npos);
    } else {
        tmp.erase(zero_pos + 1, std::string::npos);
    }
    return tmp;
}

std::vector<std::vector<int>> make_lattice(int l)
{
    std::vector<std::vector<int>> lattices;

    //cout<<"--------kx-----------"<<endl;
    for(int cell=0;cell<l*l;cell++){
        std::vector<int> s1,s2;
        int cX = cell%l;
        int cY = cell/l;

        int center = 3*cell;
        int right = 3*((cX+1)%l+l*(cY%l));

        s1 = {1+center,center};
        s2 = {right,1+center};

        lattices.push_back(s1);
        lattices.push_back(s2);
        //cout<<"{"<<s1[0]<<","<<s1[1]<<"}"<<endl;
        //cout<<"{"<<s2[0]<<","<<s2[1]<<"}"<<endl;
    }
    //cout<<endl;

    //cout<<"---------ky-----------"<<endl;
    for(int cell=0;cell<l*l;cell++){
        std::vector<int> s1,s2;
        int cX = cell%l;
        int cY = cell/l;

        int center = 3*cell;
        int up = 3*(cX%l+l*((cY+1)%l));

        s1 = {2+center,center};
        s2 = {up,2+center};

        lattices.push_back(s1);
        lattices.push_back(s2);
        //cout<<"{"<<s1[0]<<","<<s1[1]<<"}"<<endl;
        //cout<<"{"<<s2[0]<<","<<s2[1]<<"}"<<endl;
    }
    //cout<<endl;

    //cout<<"-----------kxy-----------"<<endl;
    for(int cell=0;cell<l*l;cell++){
        std::vector<int> s1,s2;
        int cX = cell%l;
        int cY = cell/l;

        int center = 3*cell;
        int cross = 3*((cX+l-1)%l+l*((cY+1)%l));

        s1 = {1+center,2+center};
        s2 = {2+center,1+cross};

        lattices.push_back(s1);
        lattices.push_back(s2);
        //cout<<"{"<<s1[0]<<","<<s1[1]<<"}"<<endl;
        //cout<<"{"<<s2[0]<<","<<s2[1]<<"}"<<endl;
    }

    return lattices;
}

