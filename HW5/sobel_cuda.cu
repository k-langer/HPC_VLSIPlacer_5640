#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include "string.h"

#define DEFAULT_THRESHOLD  4000
#define DEFAULT_FILENAME "bw_stopsign.ppm"

#include <sys/time.h>

using namespace std;
void write_ppm( char*, int, int, int, int*); 
unsigned int *read_ppm( char *, int *, int *, int *);

__global__ void sobel (int * result, unsigned int * pic, int xsize, int ysize, int thresh) {
    int j = blockIdx.x * blockDim.x + threadIdx.x; 
    int i = blockIdx.y * blockDim.y + threadIdx.y; 
    int magnitude, sum1, sum2; 
    sum1 =  pic[ xsize * (i-1) + j+1 ] -     pic[ xsize*(i-1) + j-1 ] 
        + 2 * pic[ xsize * (i)   + j+1 ] - 2 * pic[ xsize*(i)   + j-1 ]
        +     pic[ xsize * (i+1) + j+1 ] -     pic[ xsize*(i+1) + j-1 ];

    sum2 = pic[ xsize * (i-1) + j-1 ] + 2 * pic[ xsize * (i-1) + j ]  + pic[ xsize * (i-1) + j+1 ]
        - pic[xsize * (i+1) + j-1 ] - 2 * pic[ xsize * (i+1) + j ] - pic[ xsize * (i+1) + j+1 ];

    if ((sum1*sum1 + sum2*sum2)>thresh) {
        magnitude = 255;
    } else {
        magnitude = 0;
    }
    //printf("i j %d %d %d\n",i,j,pic[i*j]);
    result[i*xsize+j] = magnitude;
}

int main(int argc,char ** argv){
    int thresh = DEFAULT_THRESHOLD;
    char *filename;
    filename = strdup( DEFAULT_FILENAME);

    if (argc > 1) {
    if (argc == 3)  { // filename AND threshold
      filename = strdup( argv[1]);
       thresh = atoi( argv[2] );
    }
    if (argc == 2) { // default file but specified threshhold
      
      thresh = atoi( argv[1] );
    }

    fprintf(stderr, "file %s    threshold %d\n", filename, thresh); 
    }

    int xsize, ysize, maxval;
    unsigned int * pic = read_ppm( filename, &xsize, &ysize, &maxval ); 
    int numbytes =  xsize * ysize * 3 * sizeof( int );
    int *result = (int *) malloc( numbytes );
    if (!result) { 
        fprintf(stderr, "sobel() unable to malloc %d bytes\n", numbytes);
        exit(-1); // fail
    }
    int *out = result;

    for (int col=0; col<ysize; col++) {
        for (int row=0; row<xsize; row++) { 
          *out++ = 0; 
        }
    }

    cudaEvent_t start=0;
    cudaEvent_t stop=0;
    float time =0;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    clock_t start_st = clock(), diff;
    int *result_d;
    unsigned int *pic_d;
    
    int size = xsize*ysize*sizeof(int);
    cudaMalloc((void**)&pic_d,size);
    cudaMalloc((void**)&result_d,size);
    cudaMemcpy(result_d,result,size,cudaMemcpyHostToDevice);
    cudaMemcpy(pic_d,pic,size,cudaMemcpyHostToDevice);
    dim3 blocks(16,16);
    dim3 grid(xsize/blocks.x, ysize/blocks.y);
    cudaEventRecord(start,0);
    sobel<<<grid,blocks>>>(result_d, pic_d, xsize, ysize, thresh);
    cudaEventSynchronize(stop);
    cudaMemcpy(result,result_d,size,cudaMemcpyDeviceToHost);
    cudaEventElapsedTime(&time,start,stop);
    cudaFree(result_d);
    cudaFree(pic_d);
    diff = clock() - start_st;
    int msec = diff * 1000 / CLOCKS_PER_SEC;
    printf("Kernel time: %d s %d ms\n", msec/1000, msec%1000);
    printf("Kernel Time: %.4f\n",time);

    write_ppm( "result.ppm", xsize, ysize, 255, result);

    fprintf(stderr, "sobel done\n"); 
    return 0;
}


unsigned int *read_ppm( char *filename, int * xsize, int * ysize, int *maxval ){  
    if ( !filename || filename[0] == '\0') {
      fprintf(stderr, "read_ppm but no file name\n");
      return NULL;  // fail
    }

    FILE *fp;

    fprintf(stderr, "read_ppm( %s )\n", filename);
    fp = fopen( filename, "rb");
    if (!fp) {
      fprintf(stderr, "read_ppm()    ERROR  file '%s' cannot be opened for reading\n", filename);
      return NULL; // fail 
    }

    char chars[1024];
    //int num = read(fd, chars, 1000);
    int num = fread(chars, sizeof(char), 1000, fp);

    if (chars[0] != 'P' || chars[1] != '6') {
      fprintf(stderr, "Texture::Texture()    ERROR  file '%s' does not start with \"P6\"  I am expecting a binary PPM file\n", filename);
      return NULL;
    }

    unsigned int width, height, maxvalue;

    char *ptr = chars+3; // P 6 newline
    if (*ptr == '#') { // comment line! 
        ptr = 1 + strstr(ptr, "\n");
    }

    num = sscanf(ptr, "%d\n%d\n%d",  &width, &height, &maxvalue);
    fprintf(stderr, "read %d things   width %d  height %d  maxval %d\n", num, width, height, maxvalue);  
    *xsize = width;
    *ysize = height;
    *maxval = maxvalue;
  
    unsigned int *pic = (unsigned int *)malloc( width * height * sizeof(unsigned int));
    if (!pic) {
      fprintf(stderr, "read_ppm()  unable to allocate %d x %d unsigned ints for the picture\n", width, height);
      return NULL; // fail but return
    }

    // allocate buffer to read the rest of the file into
    int bufsize =  3 * width * height * sizeof(unsigned char);
    if ((*maxval) > 255) bufsize *= 2;
    unsigned char *buf = (unsigned char *)malloc( bufsize );
    if (!buf) {
      fprintf(stderr, "read_ppm()  unable to allocate %d bytes of read buffer\n", bufsize);
      return NULL; // fail but return
    }

    // TODO really read
    char duh[80];
    char *line = chars;

    // find the start of the pixel data.   no doubt stupid
    sprintf(duh, "%d\0", *xsize);
    line = strstr(line, duh);
    //fprintf(stderr, "%s found at offset %d\n", duh, line-chars);
    line += strlen(duh) + 1;

    sprintf(duh, "%d\0", *ysize);
    line = strstr(line, duh);
    //fprintf(stderr, "%s found at offset %d\n", duh, line-chars);
    line += strlen(duh) + 1;

    sprintf(duh, "%d\0", *maxval);
    line = strstr(line, duh);


    fprintf(stderr, "%s found at offset %ld\n", duh, line - chars);
    line += strlen(duh) + 1;

    long offset = line - chars;
    //lseek(fd, offset, SEEK_SET); // move to the correct offset
    fseek(fp, offset, SEEK_SET); // move to the correct offset
    //long numread = read(fd, buf, bufsize);
    long numread = fread(buf, sizeof(char), bufsize, fp);
    fprintf(stderr, "Texture %s   read %ld of %d bytes\n", filename, numread, bufsize); 

    fclose(fp);

    int pixels = (*xsize) * (*ysize);
    for (int i=0; i<pixels; i++) { 
        pic[i] = (int) buf[3*i];  // red channel 
    }
    return pic; // success
}

void write_ppm( char *filename, int xsize, int ysize, int maxval, int *pic) 
{
    FILE *fp;
    //never used
    //int x,y;

    fp = fopen(filename, "w");
    if (!fp) 
    {
      fprintf(stderr, "FAILED TO OPEN FILE '%s' for writing\n",filename);
      exit(-1); 
    }  

    fprintf(fp, "P6\n"); 
    fprintf(fp,"%d %d\n%d\n", xsize, ysize, maxval);

    int numpix = xsize * ysize;
    for (int i=0; i<numpix; i++) {
    unsigned char uc = (unsigned char) pic[i];
    fprintf(fp, "%c%c%c", uc, uc, uc); 
    }
    fclose(fp);
}


