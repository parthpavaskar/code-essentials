# QOL PACKAGES
using Revise, BenchmarkTools, ProgressMeter

# MATH PACKAGES
using Statistics, Distributions, FFTW, Random, FITSIO, LsqFit, Interpolations, Polynomials, LinearAlgebra, StaticArrays

# DATA PACKAGES
using HDF5, DataFrames, Images, FortranFiles, DelimitedFiles

# PLOTTING PACKAGES
using PyPlot, LaTeXStrings

using PyCall

# MULTI-THREADING
using Base
using FLoops

# CUSTOM PACKAGES
push!(LOAD_PATH, expanduser("/home/cronus/data/pavaskar/codes/code_essentials/"))
using data_struct, base_io, array_functions, turb_analysis, shockfind
