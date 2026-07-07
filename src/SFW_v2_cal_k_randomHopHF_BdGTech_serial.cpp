#include <cmath>
#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>

#include "../include/Lattice_geometry.h"
#include "../include/Eigen_system.h"
#include "../include/Equations_BdG_Tech.h"
#include "../include/argparse.h"

#define __OUTFILE__

using LATTICE = KAGOME;
using namespace BdG_Tech_RealPair;

std::vector<std::vector<int>> make_lattice(int l);
std::vector<std::pair<int,int>> make_pair(int l, std::vector<double> & Proj);
std::vector<double> make_T(const double * gamma, const std::vector<std::pair<int,int>> & pairs);

struct PROFILE
{
    const static int l=12;
    const static int L=3*l*l;
    const static int L2 = L*2;
    constexpr static double N_tot = 720;
    static std::vector<std::vector<int>> lattices;
    static double V;
};

std::vector<std::vector<int>> PROFILE::lattices = make_lattice(PROFILE::l);
double PROFILE::V = 2.0*sqrt(3.0)*PROFILE::l*PROFILE::l;

using EIGEN = EigenSetMCM<LATTICE,PROFILE>;
using EQUATION = Equations<EIGEN>;

std::string remove_zeros_in_str(const double val);

int main(int argc, char* argv[])
{
    std::vector<pair_t> options, defaults;
    // env; explanation of env
    options.push_back(pair_t("U", "interaction strength"));
    options.push_back(pair_t("X", "disorder strength"));
    options.push_back(pair_t("Tag", "Tag of a disorder sample"));
    options.push_back(pair_t("path", "directory to load and save files"));
    options.push_back(pair_t("xpath", "directory to load parameter's files"));
    // env; default value
    defaults.push_back(pair_t("X", "0"));
    defaults.push_back(pair_t("path", "."));
    // parser for arg list
    argsparse parser(argc, argv, options, defaults);

    const int L = PROFILE::L, 
        L2 = PROFILE::L2,
        l = PROFILE::l;
    const int Tag = parser.find<int>("Tag");
    const double U = parser.find<double>("U"),
        X = parser.find<double>("X"),
        V = PROFILE::V,
        N_tot = PROFILE::N_tot;
    parser.print(std::cout);

    std::vector<double> m(L), d(L), dilUsite(L,U), Proj;
    double gamma[L];
    std::fill(gamma,gamma+L,1.0);
    std::vector<std::pair<int,int>> pairs = make_pair(l,Proj);

    EIGEN eigen(L,gamma);
    EQUATION equation(eigen,L);

	std::string filename, lfilename, N_tot_str=std::to_string((int)N_tot);

    std::ifstream lfile;
    lfilename = parser.find<>("xpath") + "/KagomeProfile-L" + std::to_string(L) + "U" + parser.find<>("U") + "X" + parser.find<>("X") + "N" + N_tot_str + "-TAG" + parser.find<>("Tag") + ".dat";
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

    filename = parser.find<>("path") + "/RandomHopKagome_SFW_v2-L" + std::to_string(L) + "U" + parser.find<>("U") + "X" + parser.find<>("X") + "N" + N_tot_str + "-TAG" + parser.find<>("Tag") + ".dat"; 
#ifdef __OUTFILE__
    std::ofstream outfile;
    if (!std::ifstream(filename).is_open())
    {
        outfile.open(filename);
        outfile << "#  D^s        " << std::endl;
    }
    else {
        outfile.open(filename);
    }
    outfile.precision(10);
#endif
    eigen.set_xconf(gamma);
    std::vector<double> T = make_T(gamma,pairs);
    double sfw = equation.SFW_xx_eq_v2(m,d,V,pairs,T,Proj);
#ifdef __OUTFILE__
    outfile << "\t" << sfw << std::endl;
    outfile.close();
#else
    std::cout << "\t" << sfw << std::endl;
#endif
    return 0;
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

std::vector<std::pair<int,int>> make_pair(int l, std::vector<double> &Proj)
{
    std::vector<std::pair<int,int>> pairs;

    for(int cell=0;cell<l*l;cell++){
        std::pair<int,int> s1,s2,s3,s4,s5,s6;
        int cX = cell%l;
        int cY = cell/l;

        int center = 3*cell;
        int right = 3*((cX+1)%l+l*(cY%l));
        int cross = 3*((cX+l-1)%l+l*((cY+1)%l));
        int up = 3*(cX%l+l*((cY+1)%l));

        s1 = {1+center,center};
        s2 = {right,1+center};
        s3 = {1+center,2+center};
        s4 = {2+center,1+cross};
        s5 = {2+center,center};
        s6 = {up,2+center};

        pairs.push_back(s1);
        pairs.push_back(s2);
        pairs.push_back(s3);
        pairs.push_back(s4);
        pairs.push_back(s5);
        pairs.push_back(s6);

        Proj.push_back(1.0);
        Proj.push_back(1.0);
        Proj.push_back(0.5);
        Proj.push_back(0.5);
        Proj.push_back(0.5);
        Proj.push_back(0.5);
    }

    return pairs;
}

std::vector<double> make_T(const double * gamma, const std::vector<std::pair<int,int>> & pairs)
{
    std::vector<double> T;
    for (const auto & p : pairs) {
        T.push_back(gamma[p.first]*gamma[p.second]);
    }

    return T;
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

