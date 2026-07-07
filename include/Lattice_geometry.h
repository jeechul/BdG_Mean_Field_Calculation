#ifndef LATTICE_GEOMETRY_H
#define LATTICE_GEOMETRY_H

#include <iostream>
#include <cmath>
#include <complex>
#include <vector>
#include <map>
#include <string>

using dcomplex = std::complex<double>;

class KAGOME
{
    const double k_x,k_y;
	const int size;
	dcomplex h_p[3], h_m[3], dh_p[3], dh_m[3]; 
public:
	KAGOME(const double k_x, const double k_y, const int size) : 
    k_x(k_x), k_y(k_y),
    size(size),
    //h_* => {k_x direction, k_y direction, k_xy direction}
    h_p{dcomplex(-cos(k_x),-sin(k_x)), dcomplex(-cos(k_y),-sin(k_y)),
        dcomplex(-cos(k_x-k_y),-sin(k_x-k_y))},
    h_m{dcomplex(-cos(k_x),sin(k_x)), dcomplex(-cos(k_y),sin(k_y)),
            dcomplex(-cos(k_x-k_y),sin(k_x-k_y))},
    dh_p{dcomplex(sin(k_x),-cos(k_x)), 0., dcomplex(sin(k_x-k_y),-cos(k_x-k_y))},
    dh_m{dcomplex(sin(k_x),cos(k_x)), 0., dcomplex(sin(k_x-k_y),cos(k_x-k_y))}
    {}

    std::vector<dcomplex> make_mat(const std::vector<std::vector<int>> lattices, const bool IsNotDev)
	{
		std::vector<dcomplex> h(size*size,0);
		int arr_size = lattices.size();
		for(int l=0;l<arr_size;l++){
			int label = l/(arr_size/3);
			put_ele(&h[0],lattices[l],label,IsNotDev);
		}

		return h; 
	}

	std::vector<dcomplex> make_mat(const std::vector<std::vector<int>> lattices, const double *g, const bool IsNotDev)
	{
		std::vector<dcomplex> h(size*size,0);
		int arr_size = lattices.size();
		for(int l=0;l<arr_size;l++){
			int label = l/(arr_size/3);
			put_ele(&h[0],lattices[l],label,g,IsNotDev);
		}

		return h; 
	}

    std::vector<dcomplex> make_mat_BDM(const std::vector<std::vector<int>> lattices, const double *g, const bool IsNotDev)
    {
        std::vector<dcomplex> h(size*size,0);
        int arr_size = lattices.size();
        for(int l=0;l<arr_size;l++){
            int label = l/(arr_size/3);
            put_ele_BDM(&h[0],lattices[l],label,g,IsNotDev);
        }
        
        return h;
    }

    void make_mat_real(double * h, const std::vector<std::vector<int>> lattices, const bool IsNotDev)
	{
        unsigned long long lsize = size;
        std::fill_n(h,lsize*lsize,0.0);
		int arr_size = lattices.size();
		for(int l=0;l<arr_size;l++){
			int label = l/(arr_size/3);
			put_ele_real(&h[0],lattices[l],label,IsNotDev);
		}
	}

    void make_mat_real(double * h, const std::vector<std::vector<int>> lattices, const double *g, const bool IsNotDev)
	{
        unsigned long long lsize = size;
        std::fill_n(h,lsize*lsize,0.0);
		int arr_size = lattices.size();
		for(int l=0;l<arr_size;l++){
			int label = l/(arr_size/3);
			put_ele_real(&h[0],lattices[l],label,g,IsNotDev);
		}
	}

    void make_mat_BDM_real(double * h, const std::vector<std::vector<int>> lattices, const double *g, const bool IsNotDev)
    {
        unsigned long long lsize = size;
        std::fill_n(h,lsize*lsize,0.0);
        int arr_size = lattices.size();
        for(int l=0;l<arr_size;l++){
            int label = l/(arr_size/3);
            put_ele_BDM_real(&h[0],lattices[l],label,g,IsNotDev);
        }
    }

	std::vector<dcomplex> make_Kop(const std::vector<std::vector<int>> Kop, const char Direction)
	{
		std::vector<dcomplex> h(size*size,0);
        const bool IsNotDev = true;

		if (Direction == 'X'){ 
			int arr_size = Kop.size()/2;
			for(int l=0;l<arr_size;l++){
				put_ele(&h[0],Kop[l],0,IsNotDev);
				put_ele(&h[0],Kop[l+arr_size],2,IsNotDev);
			}
		}
		else if (Direction == 'Y'){ 
			int arr_size = Kop.size()/2;
			for(int l=0;l<arr_size;l++){
				put_ele(&h[0],Kop[l],1,IsNotDev);
				put_ele(&h[0],Kop[l+arr_size],2,IsNotDev);
			}
		}
		else if (Direction == 'D'){
			int arr_size = Kop.size();
			for(int l=0;l<arr_size;l++){
				put_ele(&h[0],Kop[l],2,IsNotDev);
			}
		}
		else{
			std::cerr<<"Wrong direction !!"<<std::endl;
			exit(1);
		} 

		return h; 
	}
private:
	void put_ele(dcomplex *h, const std::vector<int> ij, int label, const bool IsNotDev, double t=1.0)
	{
		int i = ij[0]; int j = ij[1];
		if(IsNotDev){
			h[j*size+i] = t*h_p[label];
			h[i*size+j] = t*h_m[label];	
		}
		else{
			h[j*size+i] = t*dh_p[label];
			h[i*size+j] = t*dh_m[label];
		}
	}
	
	void put_ele(dcomplex *h, const std::vector<int> ij, int label, const double *g, const bool IsNotDev) // g^{up triangle} = g^{down triangle} : with inversion symmetry
	{
		int i = ij[0]; int j = ij[1];
		if(IsNotDev){
			h[j*size+i] = g[i]*g[j]*h_p[label];
			h[i*size+j] = g[i]*g[j]*h_m[label];	
		}
		else{
			h[j*size+i] = g[i]*g[j]*dh_p[label];
			h[i*size+j] = g[i]*g[j]*dh_m[label];
		}
	}

    void put_ele_real(double *h, const std::vector<int> ij, int label, const bool IsNotDev, double t=1.0)
	{
		unsigned long long i = ij[0]; unsigned long long j = ij[1];
		if(IsNotDev){
			h[j*size+i] = t*std::real(h_p[label]);
			h[i*size+j] = t*std::real(h_m[label]);	
		}
		else{
			h[j*size+i] = t*std::real(dh_p[label]);
			h[i*size+j] = t*std::real(dh_m[label]);
		}
	}

    void put_ele_real(double *h, const std::vector<int> ij, int label, const double *g, const bool IsNotDev) // g^{up triangle} = g^{down triangle} : with inversion symmetry
	{
		unsigned long long i = ij[0]; unsigned long long j = ij[1];
		if(IsNotDev){
			h[j*size+i] = g[i]*g[j]*std::real(h_p[label]);
			h[i*size+j] = g[i]*g[j]*std::real(h_m[label]);	
		}
		else{
			h[j*size+i] = g[i]*g[j]*std::real(dh_p[label]);
			h[i*size+j] = g[i]*g[j]*std::real(dh_m[label]);
		}
	}

    void put_ele_BDM(dcomplex *h, const std::vector<int> ijs, int label, const double *g, const bool IsNotDev) // g^{up triangle} = 1/g^{down triangle} : without inversion symmetry
    {
        int i = ijs[0]; int j = ijs[1]; int s = ijs[2];
        if(IsNotDev){
            if (s == 0) { // up-triangle elements
                h[j*size+i] += g[i]*g[j]*h_p[label];
                h[i*size+j] += g[i]*g[j]*h_m[label];
            }
            else { // down-triangle elements
                h[j*size+i] += 1.0/(g[i]*g[j])*h_p[label];
                h[i*size+j] += 1.0/(g[i]*g[j])*h_m[label];
            }
        }
        else{
            if (s == 0) {
                h[j*size+i] += g[i]*g[j]*dh_p[label];
                h[i*size+j] += g[i]*g[j]*dh_m[label];
            }
            else {
                h[j*size+i] += 1.0/(g[i]*g[j])*dh_p[label];
                h[i*size+j] += 1.0/(g[i]*g[j])*dh_m[label];
            }
        }
    }

    void put_ele_BDM_real(double *h, const std::vector<int> ijs, int label, const double *g, const bool IsNotDev) // g^{up triangle} = 1/g^{down triangle} : without inversion symmetry
    {
        int i = ijs[0]; int j = ijs[1]; int s = ijs[2];
        if(IsNotDev){
            if (s == 0) { // up-triangle elements
                h[j*size+i] += g[i]*g[j]*std::real(h_p[label]);
                h[i*size+j] += g[i]*g[j]*std::real(h_m[label]);
            }
            else { // down-triangle elements
                h[j*size+i] += 1.0/(g[i]*g[j])*std::real(h_p[label]);
                h[i*size+j] += 1.0/(g[i]*g[j])*std::real(h_m[label]);
            }
        }
        else{
            if (s == 0) {
                h[j*size+i] += g[i]*g[j]*std::real(dh_p[label]);
                h[i*size+j] += g[i]*g[j]*std::real(dh_m[label]);
            }
            else {
                h[j*size+i] += 1.0/(g[i]*g[j])*std::real(dh_p[label]);
                h[i*size+j] += 1.0/(g[i]*g[j])*std::real(dh_m[label]);
            }
        }
    }
};


class KAGOMEv2
{
    const double k_x,k_y;
    const int size;
    dcomplex h_p[3], h_m[3], dhx_p[3], dhx_m[3], dhy_p[3], dhy_m[3];
public:
    KAGOMEv2(const double k_x, const double k_y, const int size) :
    k_x(k_x), k_y(k_y),
    size(size),
    //h_* => {x direction, x/2+sqrt(3)y/2 direction, x/2-sqrt(3)y/2 direction}
    h_p{dcomplex(-cos(k_x),-sin(k_x)), dcomplex(-cos(k_x/2+sqrt(3)*k_y/2),-sin(k_x/2+sqrt(3)*k_y/2)),
        dcomplex(-cos(k_x/2-sqrt(3)*k_y/2),-sin(k_x/2-sqrt(3)*k_y/2))},
    h_m{dcomplex(-cos(k_x),sin(k_x)), dcomplex(-cos(k_x/2+sqrt(3)*k_y/2),sin(k_x/2+sqrt(3)*k_y/2)),
            dcomplex(-cos(k_x/2-sqrt(3)*k_y/2),sin(k_x/2-sqrt(3)*k_y/2))},
    dhx_p{dcomplex(sin(k_x),-cos(k_x)), dcomplex(0.5*sin(k_x/2+sqrt(3)*k_y/2),-0.5*cos(k_x/2+sqrt(3)*k_y/2)), dcomplex(0.5*sin(k_x/2-sqrt(3)*k_y/2),-0.5*cos(k_x/2-sqrt(3)*k_y/2))},
    dhx_m{dcomplex(sin(k_x),cos(k_x)), dcomplex(0.5*sin(k_x/2+sqrt(3)*k_y/2),0.5*cos(k_x/2+sqrt(3)*k_y/2)), dcomplex(0.5*sin(k_x/2-sqrt(3)*k_y/2),0.5*cos(k_x/2-sqrt(3)*k_y/2))},
    dhy_p{dcomplex(0.0,0.0), dcomplex(sqrt(3)/2*sin(k_x/2+sqrt(3)*k_y/2),-sqrt(3)/2*cos(k_x/2+sqrt(3)*k_y/2)),
            dcomplex(-sqrt(3)/2*sin(k_x/2-sqrt(3)*k_y/2),sqrt(3)/2*cos(k_x/2-sqrt(3)*k_y/2))},
    dhy_m{dcomplex(0.0,0.0), dcomplex(sqrt(3)/2*sin(k_x/2+sqrt(3)*k_y/2),sqrt(3)/2*cos(k_x/2+sqrt(3)*k_y/2)),
            dcomplex(-sqrt(3)/2*sin(k_x/2-sqrt(3)*k_y/2),-sqrt(3)/2*cos(k_x/2-sqrt(3)*k_y/2))}
    {}

    std::vector<dcomplex> make_mat(const std::vector<std::vector<int>> lattices, const bool IsNotDev, const std::string dir="x")
    {
        std::vector<dcomplex> h(size*size,0);
        int arr_size = lattices.size();
        for(int l=0;l<arr_size;l++){
            int label = l/(arr_size/3);
            put_ele(&h[0],lattices[l],label,IsNotDev,dir);
        }

        return h;
    }

    std::vector<dcomplex> make_mat(const std::vector<std::vector<int>> lattices, const double *g, const bool IsNotDev, const std::string dir="x")
    {
        std::vector<dcomplex> h(size*size,0);
        int arr_size = lattices.size();
        for(int l=0;l<arr_size;l++){
            int label = l/(arr_size/3);
            put_ele(&h[0],lattices[l],label,g,IsNotDev,dir);
        }

        return h;
    }

    std::vector<dcomplex> make_mat_BDM(const std::vector<std::vector<int>> lattices, const double *g, const bool IsNotDev, const std::string dir="x")
    {
        std::vector<dcomplex> h(size*size,0);
        int arr_size = lattices.size();
        for(int l=0;l<arr_size;l++){
            int label = l/(arr_size/3);
            put_ele_BDM(&h[0],lattices[l],label,g,IsNotDev,dir);
        }

        return h;
    }

    std::vector<dcomplex> make_mat_gBDM(const std::vector<std::vector<int>> lattices, const double *g, const bool IsNotDev, const std::string dir="x")
    {
        std::vector<dcomplex> h(size*size,0);
        int arr_size = lattices.size();
        for(int l=0;l<arr_size;l++){
            int label = l/(arr_size/3);
            put_ele_gBDM(&h[0],lattices[l],label,g,IsNotDev,dir);
        }

        return h;
    }

    void make_mat_real(double * h, const std::vector<std::vector<int>> lattices, const bool IsNotDev)
    {
        unsigned long long lsize = size;
        std::fill_n(h,lsize*lsize,0.0);
        int arr_size = lattices.size();
        for(int l=0;l<arr_size;l++){
            int label = l/(arr_size/3);
            put_ele_real(&h[0],lattices[l],label,IsNotDev);
        }
    }

    void make_mat_real(double * h, const std::vector<std::vector<int>> lattices, const double *g, const bool IsNotDev)
    {
        unsigned long long lsize = size;
        std::fill_n(h,lsize*lsize,0.0);
        int arr_size = lattices.size();
        for(int l=0;l<arr_size;l++){
            int label = l/(arr_size/3);
            put_ele_real(&h[0],lattices[l],label,g,IsNotDev);
        }
    }

    void make_mat_BDM_real(double * h, const std::vector<std::vector<int>> lattices, const double *g, const bool IsNotDev)
    {
        unsigned long long lsize = size;
        std::fill_n(h,lsize*lsize,0.0);
        int arr_size = lattices.size();
        for(int l=0;l<arr_size;l++){
            int label = l/(arr_size/3);
            put_ele_BDM_real(&h[0],lattices[l],label,g,IsNotDev);
        }
    }

    void make_mat_gBDM_real(double * h, const std::vector<std::vector<int>> lattices, const double *g, const bool IsNotDev)
    {
        unsigned long long lsize = size;
        std::fill_n(h,lsize*lsize,0.0);
        int arr_size = lattices.size();
        for(int l=0;l<arr_size;l++){
            int label = l/(arr_size/3);
            put_ele_gBDM_real(&h[0],lattices[l],label,g,IsNotDev);
        }
    }

    std::vector<dcomplex> make_Kop(const std::vector<std::vector<int>> Kop, const char Direction)
    {
        std::vector<dcomplex> h(size*size,0);
        const bool IsNotDev = true;

        if (Direction == 'X'){
            int arr_size = Kop.size()/2;
            for(int l=0;l<arr_size;l++){
                put_ele(&h[0],Kop[l],0,IsNotDev);
                put_ele(&h[0],Kop[l+arr_size],2,IsNotDev);
            }
        }
        else if (Direction == 'Y'){
            int arr_size = Kop.size()/2;
            for(int l=0;l<arr_size;l++){
                put_ele(&h[0],Kop[l],1,IsNotDev);
                put_ele(&h[0],Kop[l+arr_size],2,IsNotDev);
            }
        }
        else if (Direction == 'D'){
            int arr_size = Kop.size();
            for(int l=0;l<arr_size;l++){
                put_ele(&h[0],Kop[l],2,IsNotDev);
            }
        }
        else{
            std::cerr<<"Wrong direction !!"<<std::endl;
            exit(1);
        }

        return h;
    }
private:
    void put_ele(dcomplex *h, const std::vector<int> ij, int label, const bool IsNotDev, const std::string dir="none", double t=1.0)
    {
        int i = ij[0]; int j = ij[1];
        if(IsNotDev){
            h[j*size+i] += t*h_p[label];
            h[i*size+j] += t*h_m[label];
        }
        else{
            if (dir == "x") {
                h[j*size+i] += t*dhx_p[label];
                h[i*size+j] += t*dhx_m[label];
            }
            else if (dir == "y") {
                h[j*size+i] += t*dhy_p[label];
                h[i*size+j] += t*dhy_m[label];
            }
        }
    }

    void put_ele(dcomplex *h, const std::vector<int> ij, int label, const double *g, const bool IsNotDev, const std::string dir="none") // g^{up triangle} = g^{down triangle} : with inversion symmetry
    {
        int i = ij[0]; int j = ij[1];
        if(IsNotDev){
            h[j*size+i] += g[i]*g[j]*h_p[label];
            h[i*size+j] += g[i]*g[j]*h_m[label];
        }
        else{
            if (dir == "x") {
                h[j*size+i] += g[i]*g[j]*dhx_p[label];
                h[i*size+j] += g[i]*g[j]*dhx_m[label];
            }
            else if (dir == "y") {
                h[j*size+i] += g[i]*g[j]*dhy_p[label];
                h[i*size+j] += g[i]*g[j]*dhy_m[label];
            }
        }
    }

    void put_ele_real(double *h, const std::vector<int> ij, int label, const bool IsNotDev, double t=1.0)
    {
        unsigned long long i = ij[0]; unsigned long long j = ij[1];
        if(IsNotDev){
            h[j*size+i] += t*std::real(h_p[label]);
            h[i*size+j] += t*std::real(h_m[label]);
        }
        else{
            h[j*size+i] += t*std::real(dhx_p[label]);
            h[i*size+j] += t*std::real(dhx_m[label]);
        }
    }

    void put_ele_real(double *h, const std::vector<int> ij, int label, const double *g, const bool IsNotDev) // g^{up triangle} = g^{down triangle} : with inversion symmetry
    {
        unsigned long long i = ij[0]; unsigned long long j = ij[1];
        if(IsNotDev){
            h[j*size+i] += g[i]*g[j]*std::real(h_p[label]);
            h[i*size+j] += g[i]*g[j]*std::real(h_m[label]);
        }
        else{
            h[j*size+i] += g[i]*g[j]*std::real(dhx_p[label]);
            h[i*size+j] += g[i]*g[j]*std::real(dhx_m[label]);
        }
    }

    void put_ele_BDM(dcomplex *h, const std::vector<int> ijs, int label, const double *g, const bool IsNotDev, const std::string dir="none") // g^{up triangle} = 1/g^{down triangle} : without inversion symmetry
    {
        if (ijs.size() != 3) {
            std::cerr << "# !!! ijs size < 3 !!!" << std::endl;
            exit(1);
        }

        int i = ijs[0]; int j = ijs[1]; int s = ijs[2];
        if(IsNotDev){
            if (s == 0) { // up-triangle elements
                h[j*size+i] += g[i]*g[j]*h_p[label];
                h[i*size+j] += g[i]*g[j]*h_m[label];
            }
            else { // down-triangle elements
                h[j*size+i] += 1.0/(g[i]*g[j])*h_p[label];
                h[i*size+j] += 1.0/(g[i]*g[j])*h_m[label];
            }
        }
        else{
            if (s == 0) {
                if (dir == "x"){
                    h[j*size+i] += g[i]*g[j]*dhx_p[label];
                    h[i*size+j] += g[i]*g[j]*dhx_m[label];
                }
                else if (dir == "y"){
                    h[j*size+i] += g[i]*g[j]*dhy_p[label];
                    h[i*size+j] += g[i]*g[j]*dhy_m[label];
                }
            }
            else {
                if (dir == "x"){
                    h[j*size+i] += 1.0/(g[i]*g[j])*dhx_p[label];
                    h[i*size+j] += 1.0/(g[i]*g[j])*dhx_m[label];
                }
                else if (dir == "y"){
                    h[j*size+i] += 1.0/(g[i]*g[j])*dhy_p[label];
                    h[i*size+j] += 1.0/(g[i]*g[j])*dhy_m[label];
                }
            }
        }
    }

    void put_ele_BDM_real(double *h, const std::vector<int> ijs, int label, const double *g, const bool IsNotDev) // g^{up triangle} = 1/g^{down triangle} : without inversion symmetry
    {
        if (ijs.size() != 3) {
            std::cerr << "# !!! ijs size < 3 !!!" << std::endl;
            exit(1);
        }

        int i = ijs[0]; int j = ijs[1]; int s = ijs[2];
        if(IsNotDev){
            if (s == 0) { // up-triangle elements
                h[j*size+i] += g[i]*g[j]*std::real(h_p[label]);
                h[i*size+j] += g[i]*g[j]*std::real(h_m[label]);
            }
            else { // down-triangle elements
                h[j*size+i] += 1.0/(g[i]*g[j])*std::real(h_p[label]);
                h[i*size+j] += 1.0/(g[i]*g[j])*std::real(h_m[label]);
            }
        }
        else{
            if (s == 0) {
                h[j*size+i] += g[i]*g[j]*std::real(dhx_p[label]);
                h[i*size+j] += g[i]*g[j]*std::real(dhx_m[label]);
            }
            else {
                h[j*size+i] += 1.0/(g[i]*g[j])*std::real(dhx_p[label]);
                h[i*size+j] += 1.0/(g[i]*g[j])*std::real(dhx_m[label]);
            }
        }
    }

    void put_ele_gBDM(dcomplex *h, const std::vector<int> ijs, int label, const double *g, const bool IsNotDev, const std::string dir = "none") // g^{up triangle} != g^{down triangle} : without inversion symmetry
    {
        if (ijs.size() != 3) {
            std::cerr << "# !!! ijs size < 3 !!!" << std::endl;
            exit(1);
        }

        int i = ijs[0]; int j = ijs[1]; int s = ijs[2];
        if(IsNotDev){
            if (s == 0) { // up-triangle elements
                h[j*size+i] += g[i]*g[j]*h_p[label];
                h[i*size+j] += g[i]*g[j]*h_m[label];
            }
            else { // down-triangle elements
                h[j*size+i] += g[i+size]*g[j+size]*h_p[label];
                h[i*size+j] += g[i+size]*g[j+size]*h_m[label];
            }
        }
        else{
            if (s == 0) {
                if (dir == "x"){
                    h[j*size+i] += g[i]*g[j]*dhx_p[label];
                    h[i*size+j] += g[i]*g[j]*dhx_m[label];
                }
                else if (dir == "y"){
                    h[j*size+i] += g[i]*g[j]*dhy_p[label];
                    h[i*size+j] += g[i]*g[j]*dhy_m[label];
                }
            }
            else {
                if (dir == "x"){
                    h[j*size+i] += g[i+size]*g[j+size]*dhx_p[label];
                    h[i*size+j] += g[i+size]*g[j+size]*dhx_m[label];
                }
                else if (dir == "y"){
                    h[j*size+i] += g[i+size]*g[j+size]*dhy_p[label];
                    h[i*size+j] += g[i+size]*g[j+size]*dhy_m[label];
                }
            }
        }
    }

    void put_ele_gBDM_real(double *h, const std::vector<int> ijs, int label, const double *g, const bool IsNotDev) // g^{up triangle} = 1/g^{down triangle} : without inversion symmetry
    {
        if (ijs.size() != 3) {
            std::cerr << "# !!! ijs size < 3 !!!" << std::endl;
            exit(1);
        }

        int i = ijs[0]; int j = ijs[1]; int s = ijs[2];
        if(IsNotDev){
            if (s == 0) { // up-triangle elements
                h[j*size+i] += g[i]*g[j]*std::real(h_p[label]);
                h[i*size+j] += g[i]*g[j]*std::real(h_m[label]);
            }
            else { // down-triangle elements
                h[j*size+i] += g[i+size]*g[j+size]*std::real(h_p[label]);
                h[i*size+j] += g[i+size]*g[j+size]*std::real(h_m[label]);
            }
        }
        else{
            if (s == 0) {
                h[j*size+i] += g[i]*g[j]*std::real(dhx_p[label]);
                h[i*size+j] += g[i]*g[j]*std::real(dhx_m[label]);
            }
            else {
                h[j*size+i] += g[i+size]*g[j+size]*std::real(dhx_p[label]);
                h[i*size+j] += g[i+size]*g[j+size]*std::real(dhx_m[label]);
            }
        }
    }
};


class LIEB
{
    const double k_x,k_y;
	const int size;
    const dcomplex h_p[2], h_m[2], dh_p[2], dh_m[2];
public:
	LIEB(const double k_x, const double k_y, const int size) : 
    k_x(k_x), k_y(k_y),
    size(size),
    //h_* => {k_x direction, k_y direction}
    h_p{dcomplex(-cos(k_x),-sin(k_x)),dcomplex(-cos(k_y),-sin(k_y))},
    h_m{dcomplex(-cos(k_x),sin(k_x)),dcomplex(-cos(k_y),sin(k_y))},
    dh_p{dcomplex(sin(k_x),-cos(k_x)),0.},
    dh_m{dcomplex(sin(k_x),cos(k_x)),0.}
    {}

	std::vector<dcomplex> make_mat(const std::vector<std::vector<int>> lattices, const bool IsNotDev)
	{
		std::vector<dcomplex> h(size*size,0);
		int arr_size = lattices.size();
		for(int l=0;l<arr_size;l++){
			int label = l/(arr_size/2);
			put_ele(&h[0],lattices[l],label,IsNotDev);
		}

		return h; 
	}

	std::vector<dcomplex> make_mat(const std::vector<std::vector<int>> lattices, const double *g, const bool IsNotDev)
	{
		std::vector<dcomplex> h(size*size,0);
		int arr_size = lattices.size();
		for(int l=0;l<arr_size;l++){
			int label = l/(arr_size/2);
			put_ele(&h[0],lattices[l],label,g,IsNotDev);
		}

		return h; 
	}

    std::vector<dcomplex> make_mat_tilt(const std::vector<std::vector<int>> lattices, const double eta, const bool IsNotDev)
    {
        std::vector<dcomplex> h(size*size,0);
        int arr_size = lattices.size();
        for(int l=0;l<arr_size;l++){
            int label = l/(arr_size/2);
            put_ele_tilt(&h[0],lattices[l],label,eta,IsNotDev);
        }

        return h;
    }

    void make_mat_real(double * h, const std::vector<std::vector<int>> lattices, const bool IsNotDev)
	{
        unsigned long long lsize = size;
        std::fill_n(h,lsize*lsize,0.0);
		int arr_size = lattices.size();
		for(int l=0;l<arr_size;l++){
			int label = l/(arr_size/2);
			put_ele_real(&h[0],lattices[l],label,IsNotDev);
		}
	}

    void make_mat_real(double * h, const std::vector<std::vector<int>> lattices, const double *g, const bool IsNotDev)
	{
        unsigned long long lsize = size;
        std::fill_n(h,lsize*lsize,0.0);
		int arr_size = lattices.size();
		for(int l=0;l<arr_size;l++){
			int label = l/(arr_size/2);
			put_ele_real(&h[0],lattices[l],label,g,IsNotDev);
		}
	}

    void make_mat_tilt_real(double * h, const std::vector<std::vector<int>> lattices, const double eta, const bool IsNotDev)
    {
        unsigned long long lsize = size;
        std::fill_n(h,lsize*lsize,0.0);
        int arr_size = lattices.size();
        for(int l=0;l<arr_size;l++){
            int label = l/(arr_size/2);
            put_ele_tilt_real(&h[0],lattices[l],label,eta,IsNotDev);
        }
    }

	std::vector<dcomplex> make_Kop(const std::vector<std::vector<int>> Kop, const char Direction)
	{
		std::vector<dcomplex> h(size*size,0);
        const bool IsNotDev = true;
		int dir;
		if (Direction == 'X'){ dir = 0; }
		else if (Direction == 'Y'){ dir = 1; }
		else{
			std::cerr<<"Wrong direction !!"<<std::endl;
			exit(1);
		}

		int arr_size = Kop.size();
		for(int l=0;l<arr_size;l++){
			put_ele(&h[0],Kop[l],dir,IsNotDev);
		}

		return h; 
	}
private:
	void put_ele(dcomplex *h, const std::vector<int> ij, int label, const bool IsNotDev, double t=1.0)
    {
		int i = ij[0]; int j = ij[1];
		if(IsNotDev){
			h[j*size+i] = t*h_p[label];
			h[i*size+j] = t*h_m[label];	
		}
		else{
			h[j*size+i] = t*dh_p[label];
			h[i*size+j] = t*dh_m[label];
		}
	};

	void put_ele(dcomplex *h, const std::vector<int> ij, int label, const double *g, const bool IsNotDev)
    {
		int i = ij[0]; int j = ij[1];
		if(IsNotDev){
			h[j*size+i] = g[i]*g[j]*h_p[label];
			h[i*size+j] = g[i]*g[j]*h_m[label];	
		}
		else{
			h[j*size+i] = g[i]*g[j]*dh_p[label];
			h[i*size+j] = g[i]*g[j]*dh_m[label];
		}
	}

    void put_ele_tilt(dcomplex *h, const std::vector<int> ijs, int label, const double eta, const bool IsNotDev)
    {
        int i = ijs[0]; int j = ijs[1]; int s = ijs[2];
        if(IsNotDev){
            if (s == 0) { // intra unit cell elements
                h[j*size+i] += (1+eta)*h_p[label];
                h[i*size+j] += (1+eta)*h_m[label];
            }
            else { // inter unit cell elements
                h[j*size+i] += (1-eta)*h_p[label];
                h[i*size+j] += (1-eta)*h_m[label];
            }
        }
        else{
            if (s == 0) {
                h[j*size+i] += (1+eta)*dh_p[label];
                h[i*size+j] += (1+eta)*dh_m[label];
            }
            else {
                h[j*size+i] += (1-eta)*dh_p[label];
                h[i*size+j] += (1-eta)*dh_m[label];
            }
        }
    }

    void put_ele_real(double *h, const std::vector<int> ij, int label, const bool IsNotDev, double t=1.0)
	{
		unsigned long long i = ij[0]; unsigned long long j = ij[1];
		if(IsNotDev){
			h[j*size+i] = t*std::real(h_p[label]);
			h[i*size+j] = t*std::real(h_m[label]);	
		}
		else{
			h[j*size+i] = t*std::real(dh_p[label]);
			h[i*size+j] = t*std::real(dh_m[label]);
		}
	}

    void put_ele_real(double *h, const std::vector<int> ij, int label, const double *g, const bool IsNotDev)
	{
		unsigned long long i = ij[0]; unsigned long long j = ij[1];
		if(IsNotDev){
			h[j*size+i] = g[i]*g[j]*std::real(h_p[label]);
			h[i*size+j] = g[i]*g[j]*std::real(h_m[label]);	
		}
		else{
			h[j*size+i] = g[i]*g[j]*std::real(dh_p[label]);
			h[i*size+j] = g[i]*g[j]*std::real(dh_m[label]);
		}
	}

    void put_ele_tilt_real(double *h, const std::vector<int> ijs, int label, const double eta, const bool IsNotDev)
    {
        int i = ijs[0]; int j = ijs[1]; int s = ijs[2];
        if(IsNotDev){
            if (s == 0) { // intra unit cell elements
                h[j*size+i] += (1+eta)*std::real(h_p[label]);
                h[i*size+j] += (1+eta)*std::real(h_m[label]);
            }
            else { // inter unit cell elements
                h[j*size+i] += (1-eta)*std::real(h_p[label]);
                h[i*size+j] += (1-eta)*std::real(h_m[label]);
            }
        }
        else{
            if (s == 0) {
                h[j*size+i] += (1+eta)*std::real(dh_p[label]);
                h[i*size+j] += (1+eta)*std::real(dh_m[label]);
            }
            else {
                h[j*size+i] += (1-eta)*std::real(dh_p[label]);
                h[i*size+j] += (1-eta)*std::real(dh_m[label]);
            }
        }
    }

};


class MIELKE
{
    const double k_x,k_y;
    const int size;
    const dcomplex h_p[4], h_m[4], dh_p[4], dh_m[4];
    const std::string keys[4];

    typedef std::map<std::string,std::vector<std::pair<int,int>>> LATT;

public:
    MIELKE(const double k_x, const double k_y, const int size) :
    k_x(k_x), k_y(k_y),
    size(size),
    //h_* => {k_x direction, k_y direction, k_x-k_y direction, k_x+k_y direction}
    h_p{dcomplex(-cos(k_x),-sin(k_x)), dcomplex(-cos(k_y),-sin(k_y)),
        dcomplex(-cos(k_x-k_y),-sin(k_x-k_y)), dcomplex(-cos(k_x+k_y),-sin(k_x+k_y))},
    h_m{dcomplex(-cos(k_x),sin(k_x)), dcomplex(-cos(k_y),sin(k_y)),
        dcomplex(-cos(k_x-k_y),sin(k_x-k_y)), dcomplex(-cos(k_x+k_y),sin(k_x+k_y))},
    dh_p{dcomplex(sin(k_x),-cos(k_x)), 0.,
        dcomplex(sin(k_x-k_y),-cos(k_x-k_y)), dcomplex(sin(k_x+k_y),-cos(k_x+k_y))},
    dh_m{dcomplex(sin(k_x),cos(k_x)), 0.,
        dcomplex(sin(k_x-k_y),cos(k_x-k_y)), dcomplex(sin(k_x+k_y),cos(k_x+k_y))},
    keys{"kx","ky","kx-ky","kx+ky"}
    {}

    std::vector<dcomplex> make_mat(LATT lattices, const bool IsNotDev)
    {
        std::vector<dcomplex> h(size*size,0);
        int label = 0;
        for(const auto& key : keys){
            for (const auto& connection : lattices[key]){
                put_ele(&h[0],connection,label,IsNotDev);
            }
            label++;
        }

        return h;
    }

    void make_mat_real(double * h, LATT lattices, const bool IsNotDev)
    {
        unsigned long long lsize = size;
        std::fill_n(h,lsize*lsize,0.0);
        int label = 0;
        for(const auto& key : keys){
            for (const auto& connection : lattices[key]){
                put_ele_real(h,connection,label,IsNotDev);
            }
            label++;
        }
    }

    std::vector<dcomplex> make_mat(LATT lattices, const double *g, const bool IsNotDev)
    {
        std::vector<dcomplex> h(size*size,0);
        int label = 0;
        for(const auto& key : keys){
            for (const auto& connection : lattices[key]){
                put_ele(&h[0],connection,label,g,IsNotDev);
            }
            label++;
        }

        return h;
    }

    void make_mat_real(double * h, LATT lattices, const double *g, const bool IsNotDev)
    {
        unsigned long long lsize = size;
        std::fill_n(h,lsize*lsize,0.0);
        int label = 0;
        for(const auto& key : keys){
            for (const auto& connection : lattices[key]){
                put_ele_real(h,connection,label,g,IsNotDev);
            }
            label++;
        }
    }

private:
    void put_ele(dcomplex *h, const std::pair<int,int> ij, int label, const bool IsNotDev, double t=1.0)
    {
        int i = ij.first; int j = ij.second;
        if(IsNotDev){
            h[j*size+i] = t*h_p[label];
            h[i*size+j] = t*h_m[label];
        }
        else{
            h[j*size+i] = t*dh_p[label];
            h[i*size+j] = t*dh_m[label];
        }
    }

    void put_ele(dcomplex *h, const std::pair<int,int> ij, int label, const double *g, const bool IsNotDev)
    {
        int i = ij.first; int j = ij.second;
        if(IsNotDev){
            h[j*size+i] = g[i]*g[j]*h_p[label];
            h[i*size+j] = g[i]*g[j]*h_m[label];
        }
        else{
            h[j*size+i] = g[i]*g[j]*dh_p[label];
            h[i*size+j] = g[i]*g[j]*dh_m[label];
        }
    }

    void put_ele_real(double *h, const std::pair<int,int> ij, int label, const bool IsNotDev, double t=1.0)
    {
        int i = ij.first; int j = ij.second;
        if(IsNotDev){
            h[j*size+i] = t*std::real(h_p[label]);
            h[i*size+j] = t*std::real(h_m[label]);
        }
        else{
            h[j*size+i] = t*std::real(dh_p[label]);
            h[i*size+j] = t*std::real(dh_m[label]);
        }
    }

    void put_ele_real(double *h, const std::pair<int,int> ij, int label, const double *g, const bool IsNotDev)
    {
        int i = ij.first; int j = ij.second;
        if(IsNotDev){
            h[j*size+i] = g[i]*g[j]*std::real(h_p[label]);
            h[i*size+j] = g[i]*g[j]*std::real(h_m[label]);
        }
        else{
            h[j*size+i] = g[i]*g[j]*std::real(dh_p[label]);
            h[i*size+j] = g[i]*g[j]*std::real(dh_m[label]);
        }
    }
};


#endif
