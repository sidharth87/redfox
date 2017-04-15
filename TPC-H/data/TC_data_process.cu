#include <stdlib.h>
#include <stdio.h>
#include <map>
#include <vector>
#include <set>
#include <algorithm>
#include <assert.h>
#include <string.h>

#include "Tuple.h"

#define FILE_NUM 1


#define int_bits 32

#define TC_NUM (5)

typedef Tuple<int_bits, int_bits> tc_int_t;


typedef struct tc {
    int col1;
    int col2;
} tc_t;



void read_TC() {
    char fileName[16];

    FILE *fp_in;
    FILE *fp_out;

    char * line = NULL;
    size_t len = 0;

    tc_t tc;

    int i = 0;
    tc_int_t::BasicType *tc_data = (tc_int_t::BasicType *)malloc(TC_NUM * sizeof(tc_int_t::BasicType));


    for(int fileCount=1; fileCount <= FILE_NUM; fileCount++) {
        if(FILE_NUM == 1) {
                sprintf(fileName, "TC.tbl", fileCount);
                fp_out = fopen("A", "wb");
        }
        else
                sprintf(fileName, "TC.tbl.%d", fileCount);

        fp_in = fopen(fileName, "r");

        if(!fp_in) {
            printf("Input file %s does not exist!\n", fileName);

            exit(1);
        }

        if(!fp_out) {
            printf("Input file %s does not exist!\n", "P_PARTKEY");

            exit(1);
        }


        while ((getline(&line, &len, fp_in)) != -1 && i < TC_NUM) {

            char *pch;
            pch = strtok(line, "\t");
            tc.col1 = atoi(pch);

            pch = strtok(NULL , "\t");
            tc.col2 = atoi(pch);
            
	    printf("Values are %d %d\n", tc.col1, tc.col2);
	    //std::cout << tc.col1 << tc.col2;

            //
            tc_int_t::BasicType temptc = 0;
            temptc = insert<int, 0, tc_int_t>(temptc, tc.col1);
            temptc = insert<int, 1, tc_int_t>(temptc, tc.col2);
            tc_data[i] = temptc;

            i++;
        }
    }


    fwrite(&tc_data[0], TC_NUM, sizeof(tc_int_t::BasicType), fp_out);


    if (line)
        free(line);

    fclose(fp_in);
    fclose(fp_out);
}




int main() {
    read_TC();
    return 0;
}

