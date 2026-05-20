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
    return 0;
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



int main(){

int all_ok = 1;

#define RUN_TEST(name, func) { int res = func(); print(name, res); if(!res) all_ok = 0; }

    RUN_TEST("sgemv", test_sgemv);
    RUN_TEST("dgemv", test_dgemv);
    RUN_TEST("cgemv", test_cgemv);
    RUN_TEST("zgemv", test_zgemv);
    RUN_TEST("ssymv", test_ssymv);
    RUN_TEST("dsymv", test_dsymv);
    RUN_TEST("chemv", test_chemv);
    RUN_TEST("zhemv", test_zhemv);
    RUN_TEST("strmv", test_strmv);
    RUN_TEST("dtrmv", test_dtrmv);
    RUN_TEST("ctrmv", test_ctrmv);
    RUN_TEST("ztrmv", test_ztrmv);
    RUN_TEST("strsv", test_strsv);
    RUN_TEST("dtrsv", test_dtrsv);
    RUN_TEST("sger", test_sger);
    RUN_TEST("dger", test_dger);
    RUN_TEST("cgeru", test_cgeru);
    RUN_TEST("zgeru", test_zgeru);
    RUN_TEST("cgerc", test_cgerc);
    RUN_TEST("zgerc", test_zgerc);
    RUN_TEST("ssyr", test_ssyr);
    RUN_TEST("dsyr", test_dsyr);
    RUN_TEST("cher", test_cher);
    RUN_TEST("zher", test_zher);
    RUN_TEST("ssyr2", test_ssyr2);
    RUN_TEST("dsyr2", test_dsyr2);
    RUN_TEST("cher2", test_cher2);
    RUN_TEST("zher2", test_zher2);

    return all_ok ? 0 : 1;
}