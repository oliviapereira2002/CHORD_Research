//compiling line using dynamic g++ -fPIC -o mfs.so -shared matched_filter_search.cpp -lfftw3 -lm
//compiling line g++ -fPIC -o mfs.so -shared matched_filter_search.cpp /home/hans/downloaded_code/fftw-3.3.10/.libs/libfftw3.a -lm

#include <iostream>
#include <stdio.h>
#include <math.h>       /* sqrt, fabs, log, exp */
#include <array>

#include <complex.h>
#include <fftw3.h>

#define PI float(3.14159265359)

float template_gaussian (float x, float sigma) 
{
    return 1/(sqrt(2*PI)*sigma)*std::exp(-0.5*x*x/(sigma*sigma));
}

inline void conj_multiply (fftw_complex a, fftw_complex b, fftw_complex & output)
{
    output[0] = a[0]*b[0]  + a[1]*b[1];
    output[1] = -a[0]*b[1] + a[1]*b[0];
}

void correlation (double * in1, double * in2, double * out, const unsigned int len)
{
    fftw_complex * IN1;
    fftw_complex * IN2;
    fftw_complex * conj_mult;
    IN1 = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * (len+1));
    IN2 = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * (len+1));
    conj_mult = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * (len+1));
    
    fftw_plan p1, p2, p3;
    
    p1 = fftw_plan_dft_r2c_1d(2*len, in1, IN1, FFTW_ESTIMATE);
    p2 = fftw_plan_dft_r2c_1d(2*len, in2, IN2, FFTW_ESTIMATE);
    p3 = fftw_plan_dft_c2r_1d(2*len, conj_mult, out, FFTW_ESTIMATE);
    
    fftw_execute(p1);
    fftw_execute(p2);
    
    for (unsigned int i = 0; i<len+1; i++) conj_multiply(IN1[i], IN2[i], conj_mult[i]);
    
    fftw_execute(p3);
    for (unsigned int i = 0; i < 2*len; i++) out[i] /= double(2*len); //normalizing
    
    fftw_destroy_plan(p1);
    fftw_destroy_plan(p2);
    fftw_destroy_plan(p3);
    fftw_free(IN1); fftw_free(IN2); fftw_free(conj_mult);
}

double mf_noise (const double * templ, const unsigned int N, const double original_noise)
{
    float sum = 0;
    for (unsigned int i = 0; i < N; i++)
    {
        sum += templ[i]*templ[i];
    }
    return sqrt(sum)*original_noise;
}

extern "C" {void matched_filter(const double * input_stream, const unsigned int len, const double noise, const double signal_threshold, const double width, unsigned int * detections, const unsigned int max_detections, int & n_detected, double * mfop)
{
    double * data = new double [2*len];
    for (int i = 0; i < len; i++)
    {
        data[i] = input_stream[i];
    }
    for (int i = 0; i < len; i++)
    {
        data[len+i] = 0;
    }
    
    //making gaussian template
    double * T = new double [2*len];
    for (int i = 0; i < len+1; i++) T[i] = template_gaussian(double(i), width);
    for (int i = len+1; i < 2*len; i++) T[i] = T[2*len-i];
    
    double sigma_noise = mf_noise(T, 2*len, noise);
    
    double * mf = new double [2*len];
    correlation(data,T, mf, len);
    delete[] T;
    
    double maxsnr = 0;
    for (int i = 0; i<len; i++)
    {
        if (mf[i]/sigma_noise > maxsnr) maxsnr = mf[i]/sigma_noise;
    }
    //std::cout << "Max snr:" << maxsnr << std::endl;
    
    n_detected = 0;
    unsigned int peakstart = 0xffffffff;
    for (unsigned int i = 0; i < len+1; i++)
    {
        if ((mf[i] > sigma_noise*signal_threshold) && (i < len))
        {
            if (peakstart == 0xffffffff) 
            {
                peakstart = i; 
            }
        }
        else
        {
            if (peakstart != 0xffffffff)
            {
                detections[2*n_detected] = peakstart;
                detections[2*n_detected+1] = i;
                n_detected++;
                peakstart = 0xffffffff;
            }
        }
        if (n_detected == max_detections)
        {
            std::cout << "Warning, reached size of detections array. Use higher max detections.\n";
            break;
        }
    }
    
    for (int i = 0; i<len; i++)
    {
        mfop[i] = mf[i]/sigma_noise;
    }
    
    delete[] data;
    delete[] mf;
}
}
