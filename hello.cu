#include "TMinuit.h"
#include <functional>
#include <iostream>

class FCN {
  public:
    virtual void evaluate(int&,double*,double&,double [],int) = 0;
    static void setCurrentFCN(FCN * fcn) { current = fcn; }
    static void wrapper(int& npar,double* deriv,double& f,double par[],int flag) {
      if(current) current->evaluate(npar,deriv,f,par,flag);
    }
  private:
    static FCN *current;
};

FCN *FCN::current = nullptr;

class FCN_serial: public FCN {
  public:
    FCN_serial(std::vector<double> _xs,std::vector<int> _data) : xs(_xs),data(_data) { }
    void evaluate(int& npar,double*,double& f,double par[],int) final {
      std::vector<double> result;
      for(size_t i = 0; i<data.size();++i) {
        result.push_back(this->get_LL(xs[i],data[i],par));
      }
      double sum = 0;
      for(size_t i = 0; i<data.size();++i) {
        sum += result[i];
      }
      f = sum;
    }
  private:
    double get_T(double x,double par[]) {
      return exp(par[0])+par[1]*x;
    }
    double get_LL(double x,int M,double par[]) {
      double T = this->get_T(x,par);
      return -M*log(T)+T+lgamma(M+1);
    }
    std::vector<double> xs;
    std::vector<int> data;
};

#include <thrust/transform_reduce.h>
#include <thrust/device_vector.h>
class FCN_cuda: public FCN {
  public:
    FCN_cuda(std::vector<double> _xs,std::vector<int> _data) : xs(_xs),data(_data) { }
    void evaluate(int& npar,double*,double& f,double par[],int) final {
      serial(par);
      f = parallel(par);
    }
  private:
    void serial(double par[]) {
      par[0] = exp(par[0]);
    }
    double parallel(double par[]) {
      thrust::device_vector<double> dev_x = xs;
      thrust::device_vector<int> dev_data = data;
      cuda_evalute_helper op(par[0],par[1]);
      auto begin = thrust::make_zip_iterator(thrust::make_tuple(dev_x.begin(), dev_data.begin()));
      auto end = thrust::make_zip_iterator(thrust::make_tuple(dev_x.end(), dev_data.end()));
      return thrust::transform_reduce(begin,end,op,0.,thrust::plus<double>());
    }
    std::vector<double> xs;
    std::vector<int> data;
  public:
    struct cuda_evalute_helper {
      const double p0_exp;
      const double p1;
      cuda_evalute_helper(double _p0_exp,double _p1): p0_exp(_p0_exp),p1(_p1) { }
     __device__
        double operator()(thrust::tuple<double,int> tuple) const {
          double x = thrust::get<0>(tuple);
          double M = thrust::get<1>(tuple);
          double T = p0_exp+p1*x;
          return -M*log(T)+T+lgamma(M+1);
        }
    };
};

int main() {
  std::vector<double> xs = {1,2,3,4,5,6,7};
  std::vector<int> data = {10,11,9,8,12,10,10};

  TMinuit fitter(2);
  fitter.DefineParameter(0,"p0",log(10),log(10)/100,-10,10);
  fitter.DefineParameter(1,"p1",0,1,-5,5);

  std::map<std::string,FCN*> fcns = {
    {"serial", new FCN_serial(xs,data)},
    {"cuda", new FCN_cuda(xs,data)}
  };

  for(auto fcn : fcns) {
    FCN::setCurrentFCN(fcn.second);
    fitter.SetFCN(FCN::wrapper);
    fitter.Migrad();
    double p0,p0e,p1,p1e;
    fitter.GetParameter(0,p0,p0e);
    fitter.GetParameter(1,p1,p1e);
    std::cout<<fcn.first<<" p0: "<<p0<<" ± "<<p0e<<"; p1: "<<p1<<" ± "<<p1e<<std::endl;
  }
  return 0;
}
