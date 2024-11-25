# QOL PACKAGES
using Revise, BenchmarkTools, ProgressMeter

# MATH PACKAGES
using Statistics, Distributions, FFTW, Random, FITSIO, LsqFit
using Interpolations, Polynomials, LinearAlgebra, StaticArrays

# DATA PACKAGES
using HDF5, DataFrames, Images, FortranFiles, DelimitedFiles

# PLOTTING PACKAGES
using PyPlot, LaTeXStrings, PyCall

# MULTI-THREADING
using Base
using FLoops

# CUSTOM PACKAGES
push!(LOAD_PATH, expanduser("/home/cronus/data/pavaskar/codes/code_essentials/"))
using .DataStruct, .BaseIO, .ArrayFunctions, .TurbAnalysis, .DecompMHD, .ShockFind
