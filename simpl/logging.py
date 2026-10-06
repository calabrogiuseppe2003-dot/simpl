from firedrake import PETSc

RED = "\033[1;37;31m%s\033[0m"
BLUE = "\033[1;37;34m%s\033[0m"
GREEN = "\033[1;37;32m%s\033[0m"

def info_r(message, *args, **kwargs):
    PETSc.Sys.Print(RED % message, *args, **kwargs)

def info_g(message, *args, **kwargs):
    PETSc.Sys.Print(GREEN % message, *args, **kwargs)

def info_b(message, *args, **kwargs):
    PETSc.Sys.Print(BLUE % message, *args, **kwargs)