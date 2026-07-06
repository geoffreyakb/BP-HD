#ifndef ANALYSIS_HPP_
#define ANALYSIS_HPP_

#include "idefix.hpp"
#include "input.hpp"
#include "output.hpp"
#include "grid.hpp"
#include "dataBlock.hpp"
#include "dataBlockHost.hpp"
#include <vector>
#include <string>
#include <iostream>
#include <iomanip>
#include <fstream>

class Analysis {
    public:
        Analysis(Input& input, Grid& grid, DataBlock& data);
        void ResetAnalysis();
        void PerformAnalysis(DataBlock& );

    private:
        void ComputeRadialAverage(IdefixHostArray4D<real> Vin);
        void ComputeGlobalAverage(IdefixHostArray4D<real> Vin);
        void WriteGlobalAverage(DataBlock &data);
        void WriteRadialAverage();

        DataBlockHost d;
        Grid &grid;
        GridHost gh;

        real epsilon;
        real alpha;

        int countAverage;
        int precision;
        int column_width;
        std::ofstream fileGlobalAverage;
        std::ofstream fileRadialAverage;
        std::string pathAnalysisFolder;

        int radial_NVARS;
        int Sigma;
        int Lx;
        int Ly;
        int Lz;
        int rho_Vr;
        int rho_Vperp;
        IdefixHostArray2D<real> radialAverage;
        int global_NVARS;
        int Mtot;
        IdefixHostArray1D<real> globalAverage;
};

#endif // ANALYSIS_HPP__
