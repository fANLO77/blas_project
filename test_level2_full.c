#include <stdio.h>
#include <math.h>
#include <complex.h>
#include "cblas.h"

#define EPSF 1e-5
#define EPSD 1e-9

int feq(float a, float b){ return fabsf(a-b)<EPSF; }
int deq(double a,double b){ return fabs(a-b)<EPSD; }

void print(const char* n,int ok){
    printf("%s %s\n",n,ok?"OK":"FAIL");
}

/* GEMV */

int test_sgemv(){
    float A[4]={1,2,3,4};
    float x[2]={1,1}, y[2]={0,0};
    cblas_sgemv(CblasRowMajor,CblasNoTrans,2,2,1,A,2,x,1,0,y,1);
    return feq(y[0],3)&&feq(y[1],7);
}

int test_dgemv(){
    double A[4]={1,2,3,4};
    double x[2]={1,1}, y[2]={0,0};
    cblas_dgemv(CblasRowMajor,CblasNoTrans,2,2,1,A,2,x,1,0,y,1);
    return deq(y[0],3)&&deq(y[1],7);
}

int test_cgemv(){
    float complex A[4]={1,2,3,4};
    float complex x[2]={1,1}, y[2]={0,0};
    float complex a=1,b=0;
    cblas_cgemv(CblasRowMajor,CblasNoTrans,2,2,&a,A,2,x,1,&b,y,1);
    return feq(crealf(y[0]),3)&&feq(crealf(y[1]),7);
}

int test_zgemv(){
    double complex A[4]={1,2,3,4};
    double complex x[2]={1,1}, y[2]={0,0};
    double complex a=1,b=0;
    cblas_zgemv(CblasRowMajor,CblasNoTrans,2,2,&a,A,2,x,1,&b,y,1);
    return deq(creal(y[0]),3)&&deq(creal(y[1]),7);
}

/* SYMV */

int test_ssymv(){
    float A[4]={1,2,2,3};
    float x[2]={1,1}, y[2]={0,0};
    cblas_ssymv(CblasRowMajor,CblasUpper,2,1,A,2,x,1,0,y,1);
    return feq(y[0],3)&&feq(y[1],5);
}

int test_dsymv(){
    double A[4]={1,2,2,3};
    double x[2]={1,1}, y[2]={0,0};
    cblas_dsymv(CblasRowMajor,CblasUpper,2,1,A,2,x,1,0,y,1);
    return deq(y[0],3)&&deq(y[1],5);
}

/* HEMV */

int test_chemv(){
    float complex A[4]={1,2,conjf(2),3};
    float complex x[2]={1,1}, y[2]={0,0};
    float complex a=1,b=0;
    cblas_chemv(CblasRowMajor,CblasUpper,2,&a,A,2,x,1,&b,y,1);
    return feq(crealf(y[0]),3);
}

int test_zhemv(){
    double complex A[4]={1,2,conj(2),3};
    double complex x[2]={1,1}, y[2]={0,0};
    double complex a=1,b=0;
    cblas_zhemv(CblasRowMajor,CblasUpper,2,&a,A,2,x,1,&b,y,1);
    return deq(creal(y[0]),3);
}

/* TRMV */

int test_strmv(){
    float A[4]={1,2,0,3};
    float x[2]={1,1};
    cblas_strmv(CblasRowMajor,CblasUpper,CblasNoTrans,CblasNonUnit,2,A,2,x,1);
    return feq(x[0],3)&&feq(x[1],3);
}

int test_dtrmv(){
    double A[4]={1,2,0,3};
    double x[2]={1,1};
    cblas_dtrmv(CblasRowMajor,CblasUpper,CblasNoTrans,CblasNonUnit,2,A,2,x,1);
    return deq(x[0],3)&&deq(x[1],3);
}

int test_ctrmv(){
    float complex A[4]={1,2,0,3};
    float complex x[2]={1,1};
    cblas_ctrmv(CblasRowMajor,CblasUpper,CblasNoTrans,CblasNonUnit,2,A,2,x,1);
    return feq(crealf(x[0]),3);
}

int test_ztrmv(){
    double complex A[4]={1,2,0,3};
    double complex x[2]={1,1};
    cblas_ztrmv(CblasRowMajor,CblasUpper,CblasNoTrans,CblasNonUnit,2,A,2,x,1);
    return deq(creal(x[0]),3);
}

/* TRSV */

int test_strsv(){
    float A[4]={2,1,0,3};
    float x[2]={3,3};
    cblas_strsv(CblasRowMajor,CblasUpper,CblasNoTrans,CblasNonUnit,2,A,2,x,1);
    return feq(x[1],1);
}

int test_dtrsv(){
    double A[4]={2,1,0,3};
    double x[2]={3,3};
    cblas_dtrsv(CblasRowMajor,CblasUpper,CblasNoTrans,CblasNonUnit,2,A,2,x,1);
    return deq(x[1],1);
}

/* GER */

int test_sger(){
    float A[4]={0,0,0,0};
    float x[2]={1,2}, y[2]={1,1};
    cblas_sger(CblasRowMajor,2,2,1,x,1,y,1,A,2);
    return feq(A[2],2);
}

int test_dger(){
    double A[4]={0,0,0,0};
    double x[2]={1,2}, y[2]={1,1};
    cblas_dger(CblasRowMajor,2,2,1,x,1,y,1,A,2);
    return deq(A[2],2);
}

/* GERU / GERC */

int test_cgeru(){
    float complex A[4]={0};
    float complex x[2]={1,2}, y[2]={1,1};
    float complex a=1;
    cblas_cgeru(CblasRowMajor,2,2,&a,x,1,y,1,A,2);
    return feq(crealf(A[2]),2);
}

int test_zgeru(){
    double complex A[4]={0};
    double complex x[2]={1,2}, y[2]={1,1};
    double complex a=1;
    cblas_zgeru(CblasRowMajor,2,2,&a,x,1,y,1,A,2);
    return deq(creal(A[2]),2);
}

int test_cgerc(){
    float complex A[4]={0};
    float complex x[2]={1,2}, y[2]={1,1};
    float complex a=1;
    cblas_cgerc(CblasRowMajor,2,2,&a,x,1,y,1,A,2);
    return feq(crealf(A[2]),2);
}

int test_zgerc(){
    double complex A[4]={0};
    double complex x[2]={1,2}, y[2]={1,1};
    double complex a=1;
    cblas_zgerc(CblasRowMajor,2,2,&a,x,1,y,1,A,2);
    return deq(creal(A[2]),2);
}
/* SYR / HER */

int test_ssyr(){
    float A[4]={0,0,0,0};
    float x[2]={1,2};
    cblas_ssyr(CblasRowMajor,CblasUpper,2,1,x,1,A,2);
    return feq(A[1],2) && feq(A[3],4); 
}

int test_dsyr(){
    double A[4]={0,0,0,0};
    double x[2]={1,2};
    cblas_dsyr(CblasRowMajor,CblasUpper,2,1,x,1,A,2);
    return deq(A[1],2) && deq(A[3],4);
}

int test_cher(){
    float complex A[4]={0};
    float complex x[2]={1,2};
    cblas_cher(CblasRowMajor,CblasUpper,2,1,x,1,A,2);
    return feq(crealf(A[1]),2);
}

int test_zher(){
    double complex A[4]={0};
    double complex x[2]={1,2};
    cblas_zher(CblasRowMajor,CblasUpper,2,1,x,1,A,2);
    return deq(creal(A[1]),2);
}

/* SYR2 / HER2 */

int test_ssyr2(){
    float A[4]={0};
    float x[2]={1,0}, y[2]={0,1};
    cblas_ssyr2(CblasRowMajor,CblasUpper,2,1,x,1,y,1,A,2);
    return feq(A[1],1); 
}

int test_dsyr2(){
    double A[4]={0};
    double x[2]={1,0}, y[2]={0,1};
    cblas_dsyr2(CblasRowMajor,CblasUpper,2,1,x,1,y,1,A,2);
    return deq(A[1],1);
}

int test_cher2(){
    float complex A[4]={0};
    float complex x[2]={1,0}, y[2]={0,1};
    float complex a=1;
    cblas_cher2(CblasRowMajor,CblasUpper,2,&a,x,1,y,1,A,2);
    return feq(crealf(A[1]),1);
}

int test_zher2(){
    double complex A[4]={0};
    double complex x[2]={1,0}, y[2]={0,1};
    double complex a=1;
    cblas_zher2(CblasRowMajor,CblasUpper,2,&a,x,1,y,1,A,2);
    return deq(creal(A[1]),1);
}

/* MAIN */

int main(){

print("sgemv",test_sgemv());
print("dgemv",test_dgemv());
print("cgemv",test_cgemv());
print("zgemv",test_zgemv());

print("ssymv",test_ssymv());
print("dsymv",test_dsymv());

print("chemv",test_chemv());
print("zhemv",test_zhemv());

print("strmv",test_strmv());
print("dtrmv",test_dtrmv());
print("ctrmv",test_ctrmv());
print("ztrmv",test_ztrmv());

print("strsv",test_strsv());
print("dtrsv",test_dtrsv());

print("sger",test_sger());
print("dger",test_dger());

print("cgeru",test_cgeru());
print("zgeru",test_zgeru());
print("cgerc",test_cgerc());
print("zgerc",test_zgerc());

print("ssyr",test_ssyr());
print("dsyr",test_dsyr());
print("cher",test_cher());
print("zher",test_zher());

print("ssyr2",test_ssyr2());
print("dsyr2",test_dsyr2());
print("cher2",test_cher2());
print("zher2",test_zher2());

return 0;
}