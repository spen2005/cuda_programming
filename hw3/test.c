#include <stdio.h>
#include <math.h>

#define SIZE 129 // 网格大小
#define CHARGE 1e6 // 假设点电荷的电量为1e6
#define IT 1000
#define Id 1.0

int abs(int a){
    return a>0?a:-a;
}

double potential1[SIZE+2][SIZE+2][SIZE+2]; // 用于存储电势的数组
double potential2[SIZE+2][SIZE+2][SIZE+2]; // 用于存储电势的数组

// 初始化电势数组
void initialize_potential() {
    for(int i=(SIZE+1)/2-1; i<=(SIZE+1)/2+1; i++){
        for(int j=(SIZE+1)/2-1; j<=(SIZE+1)/2+1; j++){
            for(int k=(SIZE+1)/2-1; k<=(SIZE+1)/2+1; k++){
                potential1[i][j][k] = Id/(sqrtf((i-(SIZE+1)/2)*(i-(SIZE+1)/2)+(j-(SIZE+1)/2)*(j-(SIZE+1)/2)+(k-(SIZE+1)/2)*(k-(SIZE+1)/2)));
                potential2[i][j][k] = Id/(sqrtf((i-(SIZE+1)/2)*(i-(SIZE+1)/2)+(j-(SIZE+1)/2)*(j-(SIZE+1)/2)+(k-(SIZE+1)/2)*(k-(SIZE+1)/2)));
            }
        }
    }


    for (int i = 0; i <= SIZE+1; i++) {
        for (int j = 0; j <= SIZE+1; j++) {
            for(int k=0; k<=SIZE+1; k++){
                if(abs(i-(SIZE+1)/2)<=1 && abs(j-(SIZE+1)/2)<=1 && abs(k-(SIZE+1)/2)<=1){
                    continue;
                }
                potential1[i][j][k] = 0.0;
                potential2[i][j][k] = 0.0;
            }
        }
    }
}

// 打印计算结果
void print_potential() {
    for (int i = SIZE/2+1; i <= SIZE/2+1; i++) {
        for (int j = SIZE/2+1; j <= SIZE/2+1; j++) {
            for(int k=1; k<=SIZE/2+1; k++){
                printf("%.3f ", 1/potential2[i][j][k]);
            }
            printf("\n");
        }
        printf("\n");
    }
    printf("\n\n");
    /*double identity = Id;
    for (int i = 1; i <= SIZE; i++) {
        for (int j = 1; j <= SIZE; j++) {
            for(int k=1; k<=SIZE; k++){
                printf("%.2f ", identity/potential1[i][j][k]*sqrtf((i-(SIZE+1)/2)*(i-(SIZE+1)/2)+(j-(SIZE+1)/2)*(j-(SIZE+1)/2)+(k-(SIZE+1)/2)*(k-(SIZE+1)/2)));
            }
            printf("\n");
        }
        printf("\n");
    }*/
}

// 计算网格中每个点的电势
void calculate_potential() {
    // 迭代计算电势
    int flag = 1;
    for (int iter = 0; iter < IT; iter++) { // 一般情况下需要进行多次迭代才能收敛
        double error = 0.0;
        if(flag){
            for (int i = 1; i <= SIZE; i++) {
                for (int j = 1; j <= SIZE; j++) {
                    for(int k=1; k <= SIZE; k++){
                        if(abs(i-(SIZE+1)/2)<=1 && abs(j-(SIZE+1)/2)<=1 && abs(k-(SIZE+1)/2)<=1){
                            continue;
                        }
                        int left = i - 1, right = i + 1, up = j - 1, down = j + 1, front = k - 1, back = k + 1;
                        potential2[i][j][k] = (potential1[left][j][k] + potential1[right][j][k] + potential1[i][up][k] + potential1[i][down][k] + potential1[i][j][front] + potential1[i][j][back]) / 6;
                        error += fabs(potential2[i][j][k] - potential1[i][j][k]);
                    }
                }
            }
        }
        else{
            for (int i = 1; i <= SIZE; i++) {
                for (int j = 1; j <= SIZE; j++) {
                    for(int k = 1; k <= SIZE; k++){
                        if(abs(i-(SIZE+1)/2)<=1 && abs(j-(SIZE+1)/2)<=1 && abs(k-(SIZE+1)/2)<=1){
                            continue;
                        }
                        int left = i - 1, right = i + 1, up = j - 1, down = j + 1, front = k - 1, back = k + 1;
                        potential1[i][j][k] = (potential2[left][j][k] + potential2[right][j][k] + potential2[i][up][k] + potential2[i][down][k] + potential2[i][j][front] + potential2[i][j][back]) / 6;
                        error += fabs(potential2[i][j][k] - potential1[i][j][k]);
                    }
                }
            }
        }
        flag = !flag;
        if(error < 1){
            break;
        }
    }
}

int main() {
    initialize_potential();
    calculate_potential();
    print_potential();
    return 0;
}
