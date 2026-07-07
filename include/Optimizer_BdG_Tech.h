#ifndef OPTIMIZER_BDG_TECH_H
#define OPTIMIZER_BDG_TECH_H

#include <iostream>
#include <cmath>
#include <complex>
#include <vector>
#include <algorithm>
#include <iomanip>
#include <gsl/gsl_math.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_multiroots.h>
#include <gsl/gsl_statistics.h>
#include "Equations_BdG_Tech.h"

namespace BdG_Tech_RealPair
{
/*
/////////////////////////////////////////////////////////
//---Free energy Minimize (BFGS) to find gaps & nums---//
/////////////////////////////////////////////////////////

#include "multimin_bfgs.h"

template <typename EQUATION>
class fcost 
{
private:
    EQUATION equation;
	const std::vector<double> dilute; 
	double m0;
    const int L, L2;
public:
	fcost(const EQUATION &Equation, const std::vector<double> dilute, const double m0) : 
    equation(Equation),
    dilute(dilute), m0(m0),
    L(equation.get_L()),
    L2(L*2)
	{}

	double operator()(double* p) 
	{
		std::vector<double> d(p,p+L), m(p+L,p+L2);
		return equation.W_eq(m,d,dilute,m0);
	}
	
	void df(double* p, double* g)
	{	
		std::vector<double> d(p,p+L), m(p+L,p+L2);

		double gaps[L], nums[L];

		equation.Gap_Num_eq(m,d,gaps,nums);

		//double dx = 0;
		for(int i=0;i<L;i++){
			g[i] = gaps[i]+2.*d[i]/dilute[i];
			g[i+L] = -nums[i]+2.*(m[i]-m0)/dilute[i];

			//dx += g[i]+g[i+L];
		}
		//std::cout<<"\t dx : "<<dx<<std::endl;
	}
};

template<typename EQUATION, typename PROFILE>
class Minimizer_fixedNum 
{
    EQUATION equation;
    const int L,L2;

    int minimizer(std::vector<double> &m, std::vector<double> &d, const std::vector<double> &dilUsite, double &Num, const double &m0)
    {
        using FCOST = fcost<EQUATION>;

        FCOST constrained_FreeEnergy(equation,dilUsite,m0);
        BFGS_minimizer<FCOST> multimin(constrained_FreeEnergy);

        int n = L2;	

        double result[n];
        std::copy(d.begin(),d.end(),result);
        std::copy(m.begin(),m.end(),result+L);

        //clock_t start,finish;
        //std::cout<<"....Minimize START!!"<<std::endl;
        //start = clock();

        int info = multimin.minimize(n,result);

        //finish = clock();
        //std::cout<<"....Minimize END!! : "<<(double)(finish-start)/CLOCKS_PER_SEC<<"s"<<std::endl;

        std::copy(result,result+L,d.begin());
        std::copy(result+L,result+L2,m.begin());

        double N = 0;
        //double FreeEnergy = multimin.cost(result);
        for(int i=0;i<L;i++){ 
            N += 2.*(m[i]-m0)/dilUsite[i];
            //FreeEnergy -= (m[i]-m0)*(m[i]-m0)/dilUsite[i];
        } 
        Num = N;
        //std::cout<<"....Free Energy : "<<FreeEnergy<<std::endl;

        if (info==-1) return true;
        return false;
    }
public:
    Minimizer_fixedNum(const EQUATION &Equation):
    equation(Equation),
    L(equation.get_L()),
    L2(L*2)
    {}

    int operator()(std::vector<double> &m, std::vector<double> &d, const std::vector<double> &dilUsite, double &Num, double &m0, double lambda = 0.1)
    {
        const double N = PROFILE::N_tot;
        int flag = false, iter = 0;

        double num_box[2], m0_box[2]={m0-lambda,m0+lambda}, num_mid, m0_mid, dN=1.0, dN_low, dN_hi;
        std::vector<double> m_low(m), m_hi(m), d_low(d), d_hi(d), m_mid(m), d_mid(d);
      
        minimizer(m_low,d_low,dilUsite,num_box[0],m0_box[0]);
        minimizer(m_hi,d_hi,dilUsite,num_box[1],m0_box[1]);

        while(1)
        {
            std::cout << "\r# --- num_box : " << std::setw(15) << num_box[0]<<" , "<< std::setw(15) << num_box[1] 
                << "   |   dm0 : " << std::setw(15) << (m0_box[1]-m0_box[0]) << std::flush;
            if(num_box[0] > N)
            {
                m_hi = m_low; d_hi = d_low;
                m_low = m; d_low = d;
                m0_box[1] = m0_box[0];
                m0_box[0] = m0_box[0]-2*lambda;
                num_box[1] = num_box[0];
                minimizer(m_low,d_low,dilUsite,num_box[0],m0_box[0]);
            }
            else if (num_box[1] < N)
            {
                m_low = m_hi; d_low = d_hi;
                m_hi = m; d_hi = d;
                m0_box[0] = m0_box[1];
                m0_box[1] = m0_box[1]+2*lambda;
                num_box[0] = num_box[1];
                minimizer(m_hi,d_hi,dilUsite,num_box[1],m0_box[1]);
            }
            else
            {
                break;
            }
        }

        while(1){
            std::cout << "\r# --- num_box : " << std::setw(15) << num_box[0]<<" , "<< std::setw(15) << num_box[1] 
                << "   |   dm0 : " << std::setw(15) << (m0_box[1]-m0_box[0]) << std::flush;
            
            dN_low = num_box[0]-N; dN_hi = num_box[1]-N;

            if(dN_low*dN_hi>0)
            {
                flag=true;
                std::cerr << "\r# !!! Range Error !!! \t\t\t" << std::endl;
                return flag;
            }
        
            if (std::abs(dN_low) > std::abs(dN_hi))
            { 
                dN = std::abs(dN_hi); 
            }
            else
            { 
                dN = std::abs(dN_low); 
            }
        
            if(dN<=0.01){ break; }
            else if(dN>0.3){ m0_mid = (m0_box[0]+m0_box[1])/2.; } // Bisection 
            else{ m0_mid = (m0_box[0]*dN_hi-m0_box[1]*dN_low)/(dN_hi-dN_low); } // Regula falsi

            m_mid = m; d_mid = d;	
            minimizer(m_mid,d_mid,dilUsite,num_mid,m0_mid);

            if(dN_low*(num_mid-N)<0)
            {
                m0_box[1] = m0_mid;
                m_hi = m_mid; d_hi = d_mid;
                num_box[1] = num_mid;
            }
            else
            {
                m0_box[0] = m0_mid;
                m_low = m_mid; d_low = d_mid;
                num_box[0] = num_mid;
            }

            iter++;
            if(iter>=50){
                flag = true;
                std::cerr<<"\r# !!! Over Iteration !!! \t\t\t"<<std::endl;
                return flag;
            }
        }
        std::cout<<std::endl;

        if (std::abs(dN_low) > std::abs(dN_hi))
        {
            m = m_hi; d = d_hi;
            Num = num_box[1];
            m0 = m0_box[1];
        }
        else
        {
            m = m_low; d = d_low;
            Num = num_box[0];
            m0 = m0_box[0];
        }

        return flag;
    }
};
*/

///////////////////////////////////////////////
//---Multiroot finder to find gap & chem. ---//
///////////////////////////////////////////////

template<typename EQUATION>
struct Params
{
    EQUATION Equation;
	double Num; 
	std::vector<double> dilute; 

    Params(const EQUATION &Equation_, const std::vector<double> &dilute_):
    Equation(Equation_),
    dilute(dilute_)
    {}
};

template<typename EQUATION, typename PROFILE>
int self_consist_f (const gsl_vector * x, void * params, gsl_vector * f )
{
    const std::vector<double> dilute  = ((Params<EQUATION> *) params)->dilute;
    EQUATION equation = ((Params<EQUATION> *) params)->Equation;

    const int L = equation.get_L(), L2 = L*2;

    std::vector<double> gaps(L), nums(L), d(L), m(L);
    for(int i=0;i<L;i++){
        d[i] = gsl_vector_get(x,i);
        m[i] = gsl_vector_get(x,i+L);
    } 
    double m0 = gsl_vector_get(x,L2);

    equation.Gap_Num_eq(m,d,&gaps[0],&nums[0]);

    for(int i=0;i<L;i++){
        gsl_vector_set(f,i,gaps[i]*dilute[i]/2.+d[i]);
        gsl_vector_set(f,i+L,m[i]-m0-dilute[i]*nums[i]/2.);
    }
    
    double Num = std::accumulate(nums.begin(),nums.end(),0.0);
    gsl_vector_set(f,L2,PROFILE::N_tot-Num);
    ((Params<EQUATION> *) params)->Num = Num;
    //std::cout<<"dm : "<<gsl_vector_get(f,L)<<" , dd : "<<gsl_vector_get(f,0)<<std::endl;
    
    return GSL_SUCCESS;
}

template<typename EQUATION, typename PROFILE>
class MultiRF_fixedNum
{
    EQUATION equation;
    const int L,L2;
public :
	MultiRF_fixedNum(const EQUATION &Equation):
    equation(Equation),
    L(equation.get_L()), 
    L2(L*2)
    {}
	
	int operator()(std::vector<double> &m, std::vector<double> &d, const std::vector<double> &dilUsite, double &Num, double &m0)
	{
		const gsl_multiroot_fsolver_type *T;
		gsl_multiroot_fsolver *s;
		const size_t n = L2+1;
		int status;
		size_t iter=0;

	    Params<EQUATION> p(equation,dilUsite);

		gsl_multiroot_function f = { &self_consist_f<EQUATION,PROFILE>, n, &p };
		gsl_vector * x = gsl_vector_alloc(n);

		for(int i=0;i<L;i++){
			gsl_vector_set(x,i,d[i]);
			gsl_vector_set(x,i+L,m[i]);
		}
		gsl_vector_set(x,L2,m0);

		T = gsl_multiroot_fsolver_broyden;
		s = gsl_multiroot_fsolver_alloc(T, n);
		gsl_multiroot_fsolver_set(s, &f, x);

		do
		{
			iter++;
			status = gsl_multiroot_fsolver_iterate(s);
			if (status) break;
			status = gsl_multiroot_test_residual(s->f, 1e-7);
            std::cout<<"\r# --- iter : " << std::setw(4) << iter<<" , dx : " << std::setw(17) << gsl_stats_mean(s->f->data,s->f->stride,s->f->size) << std::flush;
		}
		while (status == GSL_CONTINUE && iter < 200);
		
		for(int i = 0; i < L; i++){ 
			d[i] = gsl_vector_get(s->x, i);
			m[i] = gsl_vector_get(s->x, i+L);
		}
		Num = p.Num;
        m0 = gsl_vector_get(s->x, L2);

		gsl_multiroot_fsolver_free (s);
		gsl_vector_free (x);

		if(iter >= 200) return true;
		return false;
	}
};

/////////////////////////////////////////////
//---Relaxation Method to find gap & num---//
/////////////////////////////////////////////

template<typename EQUATION, typename PROFILE>
class Relaxation_fixedNum
{
    EQUATION equation;
    const int L,L2;

    void gap_num_consist(std::vector<double> &m, std::vector<double> &d, const std::vector<double> &dilUsite, double &Num, const double &m0)
    {
        double dm, dd, dx=1.0, nums[L], gaps[L];
        
        while(dx>=1e-8){
            equation.Gap_Num_eq(m,d,&gaps[0],&nums[0]);
            dx=0.0; dm=0.0; dd=0.0;
            for(int i=0;i<L;i++)
            {
#ifdef __NOHARTREE__
                double m_tmp = m0;
#else
                double m_tmp = m0+dilUsite[i]/2.*nums[i];
#endif
                double d_tmp = -gaps[i]*dilUsite[i]/2.;
                dm += std::abs(m[i]-m_tmp)/(double)L;
                dd += std::abs(d[i]-d_tmp)/(double)L;
                m[i] = m_tmp; d[i] = d_tmp;
            }
            dx = std::max(dm,dd);
        }
        Num = std::accumulate(nums,nums+L,0.0);
    }
public:
    Relaxation_fixedNum(const EQUATION &Equation):
    equation(Equation),
    L(equation.get_L()), 
    L2(L*2)
    {}

    int operator()(std::vector<double> &m, std::vector<double> &d, const std::vector<double> &dilUsite, double &Num, double &m0, double lambda=0.1)
    {
        const double N = PROFILE::N_tot;
        int flag = false, iter = 0;

        double num_box[2], m0_box[2]={m0-lambda,m0+lambda}, num_mid, m0_mid, dN=1.0, dN_low, dN_hi;
        std::vector<double> m_low(m), m_hi(m), d_low(d), d_hi(d), m_mid(m), d_mid(d);
        
        gap_num_consist(m_low,d_low,dilUsite,num_box[0],m0_box[0]);
        gap_num_consist(m_hi,d_hi,dilUsite,num_box[1],m0_box[1]);

        while(1)
        {
            std::cout << "\r# --- num_box : " << std::setw(15) << num_box[0]<<" , "<< std::setw(15) << num_box[1] 
                << "   |   dm0 : " << std::setw(15) << (m0_box[1]-m0_box[0]) << std::flush;
            if(num_box[0] > N)
            {
                m_hi = m_low; d_hi = d_low;
                m_low = m; d_low = d;
                m0_box[1] = m0_box[0];
                m0_box[0] = m0_box[0]-2*lambda;
                num_box[1] = num_box[0];
                gap_num_consist(m_low,d_low,dilUsite,num_box[0],m0_box[0]);
            }
            else if (num_box[1] < N)
            {
                m_low = m_hi; d_low = d_hi;
                m_hi = m; d_hi = d;
                m0_box[0] = m0_box[1];
                m0_box[1] = m0_box[1]+2*lambda;
                num_box[0] = num_box[1];
                gap_num_consist(m_hi,d_hi,dilUsite,num_box[1],m0_box[1]);
            }
            else
            {
                break;
            }
        }

        while(1)
        {
            std::cout << "\r# --- num_box : " << std::setw(15) << num_box[0]<<" , "<< std::setw(15) << num_box[1] 
                << "   |   dm0 : " << std::setw(15) << (m0_box[1]-m0_box[0]) << std::flush;
        
            dN_low = num_box[0]-N; dN_hi = num_box[1]-N;
            if(dN_low*dN_hi>0){
                flag=true;
                std::cerr << "\r# !!! Range Error !!! \t\t\t" << std::endl;
                return flag;
            }
            
            if(std::abs(dN_low) > std::abs(dN_hi)) 
                dN = std::abs(dN_hi);
            else 
                dN = std::abs(dN_low);

            if(dN<=0.01){ break; }
            else if(dN>0.3){ m0_mid = (m0_box[0]+m0_box[1])/2.; } // Bisection
            else{ m0_mid = (m0_box[0]*dN_hi-m0_box[1]*dN_low)/(dN_hi-dN_low); } // Regula falsi 

            m_mid = m; d_mid = d;
            gap_num_consist(m_mid,d_mid,dilUsite,num_mid,m0_mid);

            if(dN_low*(num_mid-N)<0){
                m0_box[1] = m0_mid;
                m_hi = m_mid; d_hi = d_mid;
                num_box[1] = num_mid;
            }
            else{
                m0_box[0] = m0_mid;
                m_low = m_mid; d_low = d_mid;
                num_box[0] = num_mid;
            }

            iter++;
            if(iter>=50){
                flag = true;
                std::cerr << "\r# !!! Over Iteration !!! \t\t\t" << std::endl;
                return flag;
            }
        }
        std::cout<<std::endl;

        if(std::abs(dN_low) > std::abs(dN_hi))
        {
            m = m_hi; d = d_hi;
            Num = num_box[1];
            m0 = m0_box[1];
        }
        else
        {
            m = m_low; d = d_low;
            Num = num_box[0];
            m0 = m0_box[0];
        }

        return flag;
    }
};

template<typename EQUATION, typename PROFILE>
class Relaxation_fixedNum_v2
{
    EQUATION equation;
    const int L,L2;

    void gap_num_consist(std::vector<double> &m, std::vector<double> &gaps, const std::vector<double> &dilUsite, double &Num, const double &m0)
    {
        double dm, dd, dx=1.0, nums[L];
        std::vector<double> d(L);
        for (int i=0;i<L;++i) d[i] = -gaps[i]*dilUsite[i]/2.;
        
        while(dx>=1e-8){
            equation.Gap_Num_eq(m,d,&gaps[0],&nums[0]);
            dx=0.0; dm=0.0; dd=0.0;
            for(int i=0;i<L;i++)
            {
#ifdef __NOHARTREE__
                double m_tmp = m0;
#else
                double m_tmp = m0+dilUsite[i]/2.*nums[i];
#endif
                double d_tmp = -gaps[i]*dilUsite[i]/2.;
                dm += std::abs(m[i]-m_tmp)/(double)L;
                dd += std::abs(d[i]-d_tmp)/(double)L;
                m[i] = m_tmp; d[i] = d_tmp;
            }
            dx = std::max(dm,dd);
        }
        Num = std::accumulate(nums,nums+L,0.0);
    }
public:
    Relaxation_fixedNum_v2(const EQUATION &Equation):
    equation(Equation),
    L(equation.get_L()), 
    L2(L*2)
    {}

    int operator()(std::vector<double> &m, std::vector<double> &gaps, const std::vector<double> &dilUsite, double &Num, double &m0, double lambda=0.1)
    {
        const double N = PROFILE::N_tot;
        int flag = false, iter = 0;

        double num_box[2], m0_box[2]={m0-lambda,m0+lambda}, num_mid, m0_mid, dN=1.0, dN_low, dN_hi;
        std::vector<double> m_low(m), m_hi(m), gaps_low(gaps), gaps_hi(gaps), m_mid(m), gaps_mid(gaps);
        
        gap_num_consist(m_low,gaps_low,dilUsite,num_box[0],m0_box[0]);
        gap_num_consist(m_hi,gaps_hi,dilUsite,num_box[1],m0_box[1]);

        while(1)
        {
            std::cout << "\r# --- num_box : " << std::setw(15) << num_box[0]<<" , "<< std::setw(15) << num_box[1] 
                << "   |   dm0 : " << std::setw(15) << (m0_box[1]-m0_box[0]) << std::flush;
            if(num_box[0] > N)
            {
                m_hi = m_low; gaps_hi = gaps_low;
                m_low = m; gaps_low = gaps;
                m0_box[1] = m0_box[0];
                m0_box[0] = m0_box[0]-2*lambda;
                num_box[1] = num_box[0];
                gap_num_consist(m_low,gaps_low,dilUsite,num_box[0],m0_box[0]);
            }
            else if (num_box[1] < N)
            {
                m_low = m_hi; gaps_low = gaps_hi;
                m_hi = m; gaps_hi = gaps;
                m0_box[0] = m0_box[1];
                m0_box[1] = m0_box[1]+2*lambda;
                num_box[0] = num_box[1];
                gap_num_consist(m_hi,gaps_hi,dilUsite,num_box[1],m0_box[1]);
            }
            else
            {
                break;
            }
        }

        while(1)
        {
            std::cout << "\r# --- num_box : " << std::setw(15) << num_box[0]<<" , "<< std::setw(15) << num_box[1] 
                << "   |   dm0 : " << std::setw(15) << (m0_box[1]-m0_box[0]) << std::flush;
        
            dN_low = num_box[0]-N; dN_hi = num_box[1]-N;
            if(dN_low*dN_hi>0){
                flag=true;
                std::cerr << "\r# !!! Range Error !!! \t\t\t" << std::endl;
                return flag;
            }
            
            if(std::abs(dN_low) > std::abs(dN_hi)) 
                dN = std::abs(dN_hi);
            else 
                dN = std::abs(dN_low);

            if(dN<=0.01){ break; }
            else if(dN>0.3){ m0_mid = (m0_box[0]+m0_box[1])/2.; } // Bisection
            else{ m0_mid = (m0_box[0]*dN_hi-m0_box[1]*dN_low)/(dN_hi-dN_low); } // Regula falsi 

            m_mid = m; gaps_mid = gaps;
            gap_num_consist(m_mid,gaps_mid,dilUsite,num_mid,m0_mid);

            if(dN_low*(num_mid-N)<0){
                m0_box[1] = m0_mid;
                m_hi = m_mid; gaps_hi = gaps_mid;
                num_box[1] = num_mid;
            }
            else{
                m0_box[0] = m0_mid;
                m_low = m_mid; gaps_low = gaps_mid;
                num_box[0] = num_mid;
            }

            iter++;
            if(iter>=50){
                flag = true;
                std::cerr << "\r# !!! Over Iteration !!! \t\t\t" << std::endl;
                return flag;
            }
        }
        std::cout<<std::endl;

        if(std::abs(dN_low) > std::abs(dN_hi))
        {
            m = m_hi; gaps = gaps_hi;
            Num = num_box[1];
            m0 = m0_box[1];
        }
        else
        {
            m = m_low; gaps = gaps_low;
            Num = num_box[0];
            m0 = m0_box[0];
        }

        return flag;
    }
};

} // namespace BdG_Tech_RealPair


namespace BdG_Tech_ComplexPair
{

/////////////////////////////////////////////
//---Relaxation Method to find gap & num---//
/////////////////////////////////////////////

template<typename EQUATION, typename PROFILE>
class Relaxation_fixedNum
{
    EQUATION equation;
    const int L,L2;

    void gap_num_consist(std::vector<double> &m, std::vector<dcomplex> &d, const std::vector<double> &dilUsite, double &Num, const double &m0)
    {
        double dm, dd, dx=1.0, nums[L]; 
        dcomplex gaps[L];
        
        while(dx>=1e-8){
            equation.Gap_Num_eq(m,d,&gaps[0],&nums[0]);
            dx=0.0; dm=0.0; dd=0.0;
            for(int i=0;i<L;i++)
            {
#ifdef __NOHARTREE__
                double m_tmp = m0;
#else
                double m_tmp = m0+dilUsite[i]/2.*nums[i];
#endif
                dcomplex d_tmp = -gaps[i]*dilUsite[i];
                dm += std::abs(m[i]-m_tmp)/(double)L;
                dd += std::abs(std::abs(d[i])-std::abs(d_tmp))/(double)L;
                m[i] = m_tmp; d[i] = d_tmp;
            }
            dx = std::max(dm,dd);
            //std::cout<<dx<<"\t"<<dm<<"\t"<<dd<<std::endl;
        }
        Num = std::accumulate(nums,nums+L,0.0);
    }
public:
    Relaxation_fixedNum(const EQUATION &Equation):
    equation(Equation),
    L(equation.get_L()), 
    L2(L*2)
    {}

    int operator()(std::vector<double> &m, std::vector<dcomplex> &d, const std::vector<double> &dilUsite, double &Num, double &m0, double lambda=0.1)
    {
        const double N = PROFILE::N_tot;
        int flag = false, iter = 0;

        double num_box[2], m0_box[2]={m0-lambda,m0+lambda}, num_mid, m0_mid, dN=1.0, dN_low, dN_hi;
        std::vector<double> m_low(m), m_hi(m), m_mid(m);
        std::vector<dcomplex> d_low(d), d_hi(d), d_mid(d);
        
        gap_num_consist(m_low,d_low,dilUsite,num_box[0],m0_box[0]);
        gap_num_consist(m_hi,d_hi,dilUsite,num_box[1],m0_box[1]);

        while(1)
        {
            std::cout << "\r# --- num_box : " << std::setw(15) << num_box[0]<<" , "<< std::setw(15) << num_box[1] 
                << "   |   dm0 : " << std::setw(15) << (m0_box[1]-m0_box[0]) << std::flush;
            if(num_box[0] > N)
            {
                m_hi = m_low; d_hi = d_low;
                m_low = m; d_low = d;
                m0_box[1] = m0_box[0];
                m0_box[0] = m0_box[0]-2*lambda;
                num_box[1] = num_box[0];
                gap_num_consist(m_low,d_low,dilUsite,num_box[0],m0_box[0]);
            }
            else if (num_box[1] < N)
            {
                m_low = m_hi; d_low = d_hi;
                m_hi = m; d_hi = d;
                m0_box[0] = m0_box[1];
                m0_box[1] = m0_box[1]+2*lambda;
                num_box[0] = num_box[1];
                gap_num_consist(m_hi,d_hi,dilUsite,num_box[1],m0_box[1]);
            }
            else
            {
                break;
            }
        }

        while(1)
        {
            std::cout << "\r# --- num_box : " << std::setw(15) << num_box[0]<<" , "<< std::setw(15) << num_box[1] 
                << "   |   dm0 : " << std::setw(15) << (m0_box[1]-m0_box[0]) << std::flush;
        
            dN_low = num_box[0]-N; dN_hi = num_box[1]-N;

            if(dN_low*dN_hi>0){
                flag=true;
                std::cerr << "\r# !!! Range Error !!! \t\t\t" << std::endl;
                return flag;
            }
            
            if(std::abs(dN_low) > std::abs(dN_hi))
            {
                dN = std::abs(dN_hi);
            }
            else 
            {
                dN = std::abs(dN_low);
            }

            if(dN<=0.01){ break; }
            else if(dN>0.3){ m0_mid = (m0_box[0]+m0_box[1])/2.; } // Bisection
            else{ m0_mid = (m0_box[0]*dN_hi-m0_box[1]*dN_low)/(dN_hi-dN_low); } // Regula falsi 

            m_mid = m; d_mid = d;
            gap_num_consist(m_mid,d_mid,dilUsite,num_mid,m0_mid);

            if(dN_low*(num_mid-N)<0)
            {
                m0_box[1] = m0_mid;
                m_hi = m_mid; d_hi = d_mid;
                num_box[1] = num_mid;
            }
            else
            {
                m0_box[0] = m0_mid;
                m_low = m_mid; d_low = d_mid;
                num_box[0] = num_mid;
            }

            iter++;
            if(iter>=50){
                flag = true;
                std::cerr << "\r# !!! Over Iteration !!! \t\t\t" << std::endl;
                return flag;
            }
        }
        std::cout<<std::endl;

        if(std::abs(dN_low) > std::abs(dN_hi))
        {
            m = m_hi; d = d_hi;
            Num = num_box[1];
            m0 = m0_box[1];
        }
        else
        {
            m = m_low; d = d_low;
            Num = num_box[0];
            m0 = m0_box[0];
        }

        return flag;
    }
};

} // namespace BdG_Tech_ComplexPair

#endif
