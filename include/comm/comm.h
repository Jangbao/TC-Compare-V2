#pragma once

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <sys/types.h>
#include <unistd.h>

#include <fstream>
#include <iostream>

#include "ini/INIReader.h"
#include "spdlog/spdlog.h"

#define MEMORY_K (1024)
#define MEMORY_M (1024 * 1024)
#define MEMORY_G (1024 * 1024 * 1024)

typedef unsigned int count_t;
typedef unsigned long int index_t;
typedef unsigned int vertex_t;

typedef struct edge {
    index_t u, v;
} edge;

typedef struct vertex {
    index_t id;
    uint degree;
} vertex;

inline off_t fsize(const char *filename) {
    struct stat st;
    if (stat(filename, &st) == 0) {
        return st.st_size;
    }
    return -1;
}

inline double wtime() {
    double time[2];
    struct timeval time1;
    gettimeofday(&time1, NULL);

    time[0] = time1.tv_sec;
    time[1] = time1.tv_usec;

    return time[0] + time[1] * 1.0e-6;
}
