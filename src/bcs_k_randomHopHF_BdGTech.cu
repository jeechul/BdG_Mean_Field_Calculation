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

//#define __NOHARTREE__
#define __GPU__
#include "../include/Lattice_geometry.h"
#include "../include/Eigen_system.h"
#include "../include/Equations_BdG_Tech.h"
#include "../include/Optimizer_BdG_Tech.h"
#include "../include/argparse.h"

#define __OUTFILE__

using LATTICE = KAGOME;
using namespace BdG_Tech_RealPair;

std::vector<std::vector<int>> make_lattice(int l);

struct PROFILE
{
    const static int l=12;
    const static int L=3*l*l;
    const static int L2=L*2;
    constexpr static double N_tot = 720; // # of Total electrons
    static std::vector<std::vector<int>> lattices;
};

std::vector<std::vector<int>> PROFILE::lattices = make_lattice(PROFILE::l);

using EIGEN = EigenSetMCM<LATTICE,PROFILE>;
using EQUATION = Equations<EIGEN>;
using OPTIMIZER = Relaxation_fixedNum<EQUATION,PROFILE>;
using OPTIMIZER0 = MultiRF_fixedNum<EQUATION,PROFILE>;

std::string remove_zeros_in_str(const double val);

int main(int argc, char* argv[])
{
    std::vector<pair_t> options, defaults;
    // env; explanation of env
    options.push_back(pair_t("U", "interaction strength"));
    options.push_back(pair_t("mi", "initial guess of chemical potential"));
    options.push_back(pair_t("di", "initial guess of gap"));
    options.push_back(pair_t("Nsam", "# of disorder samples"));
    options.push_back(pair_t("Tag0", "Initial tag number of disorder samples"));
    options.push_back(pair_t("X", "disorder strength"));
    options.push_back(pair_t("dev", "device number"));
    options.push_back(pair_t("lambda", "hyper parameter to tune total Num"));
    options.push_back(pair_t("IsReInit", "turn on Re-initializing mi & di"));
    options.push_back(pair_t("path", "directory to load and save files"));
    // env; default value
    defaults.push_back(pair_t("Nsam", "1"));
    defaults.push_back(pair_t("Tag0", "0"));
    defaults.push_back(pair_t("X", "0.0"));
    defaults.push_back(pair_t("lambda", "0.1"));
    defaults.push_back(pair_t("IsReInit", "1"));
    defaults.push_back(pair_t("path", "."));
    // parser for arg list
    argsparse parser(argc, argv, options, defaults);

    const int L = PROFILE::L,
        L2 = PROFILE::L2,
        l = PROFILE::l;
    const int Nsam = parser.find<int>("Nsam"),
        Tag0 = parser.find<int>("Tag0"),
        IsReInit = parser.find<int>("IsReInit"),
        dev = parser.find<int>("dev");
    const double U = parser.find<double>("U"),
        mi = parser.find<double>("mi"),
        di = parser.find<double>("di"),
        lambda = parser.find<double>("lambda"),
        X = parser.find<double>("X"),
        N_tot = PROFILE::N_tot;
    parser.print(std::cout);

    cudaSetDevice(dev);

    std::random_device rn;
    std::mt19937 generator(rn());
    std::uniform_real_distribution<double> dist(-1.0,1.0);

    double gamma[L];

    EIGEN eigen(L,gamma);
    EQUATION equation(eigen,L);
    OPTIMIZER optimizer(equation);
    OPTIMIZER0 optimizer0(equation); 

    double n,mg,mi_,di_;
    std::vector<double> m(L,mi), d(L,di), dilUsite(L,U);
    std::fill(gamma,gamma+L,1.0);
    if (IsReInit)
    {
        mg = mi-U/2.*N_tot/(double)L;
        optimizer0(m,d,dilUsite,n,mg);
        mi_ = m[0]; di_ = d[0];
        std::cout << "\r#   ---   Re-initialization Finish   ---   #" << std::endl;
    }
    else
    {
        n = N_tot;
        mi_ = mi;
        di_ = di;
    }

    for (int i=Tag0; i<Tag0+Nsam; ++i)
    {
        int flag = false;
        mg = mi_-U/2.*n/(double)L; // initial global chemical potential

        for(int j=0;j<L;++j) 
            gamma[j] = 1.0+X*dist(generator);

        eigen.set_xconf(gamma);

        std::string N_tot_str = std::to_string((int)N_tot); //remove_zeros_in_str(N_tot);
        std::string filename = parser.find<>("path") + "/KagomeProfile-L" + std::to_string(L) + "U" + parser.find<>("U") + "X" + parser.find<>("X") + "N" + N_tot_str + "-TAG" + std::to_string(i) + ".dat"; 
#ifdef __OUTFILE__
        std::ofstream outfile;
        if (!std::ifstream(filename).is_open())
        {
            outfile.open(filename);
            outfile << "#       X\tY\tV\t<c_up*c_dn>\t<n_up+n_dn>\tMU\tMU0     " << std::endl;
            std::cout << "#--- Estimating ... " << std::setw(20) << filename << std::endl;
        }
        else
        { 
            continue; 
        }
        outfile.precision(10);
#else 
        std::cout << "#       X\tY\tV\t<c_up*c_dn>\t<n_up+n_dn>\tMU\tMU0     " << std::endl;
#endif
        clock_t start = clock();
        flag = optimizer(m,d,dilUsite,n,mg,lambda);
        std::cout << "\n# duration time : " << (clock()-start)/(double)CLOCKS_PER_SEC << " s" << std::endl;
#ifdef __OUTFILE__
        for (int j=0;j<L;++j)
        {
            int supX = (j/3)%l;
            int supY = (j/3)/l;
            int subID = (j%3);

            double x = supX+supY*0.5, y = supY*std::sqrt(3.0)/2.0;
            if (subID == 1) { 
                x += 0.5;
            }    
            else if (subID == 2) {
                x += 0.25;
                y += std::sqrt(3.0)/4.0;
            }

            outfile << "\t" << x << "\t" << y << "\t" << gamma[j] << "\t"
                << d[j]/dilUsite[j] << "\t" << 2*(m[j]-mg)/dilUsite[j] << "\t" << m[j] << "\t" << mg <<std::endl;
        }
        outfile << "# Ntot\t" << n << std::endl;
#else
        for (int j=0;j<L;++j)
        {
            int supX = (j/3)%l;
            int supY = (j/3)/l;
            int subID = (j%3);

            double x = supX+supY*0.5, y = supY*std::sqrt(3.0)/2.0;
            if (subID == 1) { 
                x += 0.5;
            }    
            else if (subID == 2) {
                x += 0.25;
                y += std::sqrt(3.0)/4.0;
            }

            std::cout << "\t" << x << "\t" << y << "\t" << gamma[j] << "\t"
                << d[j]/dilUsite[j] << "\t" << 2*(m[j]-mg)/dilUsite[j] << "\t" << m[j] << "\t" << mg <<std::endl;
        }
        std::cout << "# Ntot\t" << n << std::endl;
#endif
        m.assign(L,mi_);
        d.assign(L,di_);
    }

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

