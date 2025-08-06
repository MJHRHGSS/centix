#include <stdio.h>
#include <string.h>
int logcx = 0;
FILE *logfile = NULL;
int iscxf(char *f) {
    if (!f) return 0;
    size_t len = strlen(f);
    if (len < 3) return 0;
    return strcmp(f + (len - 3), ".cx") == 0;    
}
int main(int argc, char **argv) {
    if (argc < 2 || argc > 4) {
        printf("Usage: cx [file.cx] [--log file.log] [--version] [--help]\n");
        return 1;
    }
    if (strcmp(argv[1], "--version") == 0) {
        printf("cx (Centix) Alpha-1.0.0\nThis is free software; see the source for copying conditions. There is NO\nwarranty; not even for MERCHANTABILTY or FITNESS FOR A PARTICULAR PURPOSE\n\n");
        return 0;
    } else if (strcmp(argv[1], "--help") == 0) {
        printf("Options:\n\t--help\t\tDisplay this\n\t--version\tDisplay interpreter version information\n\t--log\t\tlog the file to a specific location\n");
        return 0;
    }
    if (!iscxf(argv[1])) {
        printf("Specified file is not a Centix file, please make sure you are using the `.cx' extention\n");
        return 1;
    }
    if (argc == 4) {
        logcx = 1;
        logfile = fopen(argv[3], "w");
        if (!logfile) {
            perror("open");
            return 1;
        }
    }
    if (logcx) fclose(logfile);
    return 0;
}
