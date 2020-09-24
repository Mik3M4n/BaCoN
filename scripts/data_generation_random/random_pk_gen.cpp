#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <sys/stat.h>
#include <cmath>


#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <time.h>
#include <cmath>
#include <random>


using namespace std;
/* Generate random spectra from filters and some reference spectra */

// Random integer generator in range fMin<n<fMax
static int fRand(double fMin, double fMax)
{
    double f = (double)rand() / RAND_MAX;
        return round(fMin + f * (fMax - fMin));
}


int main(int argc, char* argv[]) {

  srand (pow(time(NULL),2)+time(NULL));

// Number of filters to be used and number of reference spectra per class - see random_pk/ref_pk
int Nfilters =500;
int noref = 2500;

for(int i = 1; i<= Nfilters; i++){
  vector<vector<double> > refpk,filter;
  // Get reference spectrum randomly
  char pofk[41];
  // pofk top directory
  string abcd ="random_pk/ref_pk/";

  string model;
  // Select model (LCDM,wCDM,fR,DGP)
  int mymodel = fRand(1, 4.5);

  switch (mymodel) {
    case 1:
        model ="lcdm/";
          break;
    case 2:
        model ="wcdm/";
          break;
    case 3:
        model ="fr/";
          break;
    case 4:
        model ="dgp/";
          break;
    default:
    printf("invalid indices, mymodel = %d \n", mymodel);
        return 0;
  }
  // random index from the random theory
  int rind = fRand(1, noref);

  sprintf(pofk, "%s%s%i%s",abcd.c_str(),model.c_str(),rind,".txt");

ifstream fin(pofk);
string line;
while (getline(fin, line)) {      // for each line
        vector<double> lineData;           // create a new row
        double val;
        istringstream lineStream(line);
        while (lineStream >> val) {          // for each value in line
                lineData.push_back(val);           // add to the current row
        }
        refpk.push_back(lineData);         // add row to allData
}

// Get filter sequentially
/*output name */
char filtername[41];
// output file directory
string abc ="random_pk/filters/";
sprintf(filtername, "%s%i%s",abc.c_str(),i,".txt");

ifstream fin2(filtername);
string line2;
while (getline(fin2, line2)) {      // for each line
        vector<double> lineData;           // create a new row
        double val;
        istringstream lineStream(line2);
        while (lineStream >> val) {          // for each value in line
                lineData.push_back(val);           // add to the current row
        }
        filter.push_back(lineData);         // add row to allData
}

printf("%d %d %s %d \n", i, mymodel, model.c_str() , rind);


/*output name */
char output[41];
// output file directory
string ab ="random_pk/rand_pk/";
sprintf(output, "%s%i%s",ab.c_str(),i,".txt");

/* Open output file */
FILE* fp = fopen(output, "w");


for(int j =0; j < refpk.size();  j ++) {

    double mydata[4];
    for(int k=0;k<4;k++){
    mydata[k] = refpk[j][k+1]*filter[j][k+1];
    }
    fprintf(fp,"%e %e %e %e %e \n", refpk[j][0], mydata[0], mydata[1], mydata[2], mydata[3]); // print to file

  }

  /*close output file*/
    fclose(fp);

}

    return 0;
}
