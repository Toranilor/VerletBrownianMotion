/*
A version of my 'Major Verlet 2D Speedy' with Eigen classes instead.
I have replaced only the vectors/matrices that are used in actual data operations


Compiler Command: g++ -I C:/Eigen -O3 MajorVerletEigen.cpp -o MajorVerletEigen
*/

#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>
#include <random>
#include <chrono>
#include <vector>
#include <string>
#include <assert.h>
#include <cstdlib>
#include <unordered_map>
#include <algorithm>
#include <cstdint>
#include <iostream>
#include <Eigen/Dense>


using namespace std;
using Eigen::MatrixXd;

#define PI 3.14159265359 


void print_array(MatrixXd &im_array){
    // A function that prints the array to screen
    for(int i=0; i<im_array.rows(); i++){
        for(int j=0; j<im_array.cols(); j++){
            cout<<im_array(i,j)<<' ';
        }
        cout<<endl;
    }
}

unordered_map<string, string> config_parse(string file_loc){
    unordered_map<string, string> data; // Create an unordered map
    ifstream file(file_loc); // Generate a new file
    string line;            // Create a new vairable called line
    while (getline(file,line))
    {
        if (line.find('=') != string::npos)
        {
            // We have found a match to our delimeter
            stringstream line_ss(line);
            string segment;
            vector<string> vec;
            while(getline(line_ss, segment, '='))
            {
                cout<<segment<<'\n';
                vec.push_back(segment);
            }
            data[vec.at(0)] = vec.at(1);
        }

    }
    return data;
}

double a_step (double vt, double drag, double mass, double chi, double fext, double rand_term)
{
    
    double at_dt;
    at_dt = (chi*rand_term + fext - drag*vt)/mass;
    return at_dt;
}

double v_step (double vt, double at, double at_dt, double dt)
{
    double v_step;
    v_step = vt + 0.5*(at+at_dt)*dt;
    return v_step;
}

double r_step (double rt, double vt, double at, double dt)
{
    double rt_dt;
    rt_dt = rt + vt*dt + 0.5*at*dt*dt;
    return rt_dt;
}

double f_get(double stiffness, double rt)
{
    double f_out;
    f_out = (-1)*stiffness*rt;
    return f_out;
}

double i_get(MatrixXd &im_array, double xt, double yt,
            double x_width, double y_width)
{
    // This function returns the intensity at point rt
    // Intensity is given by 2D vector array img_array
    // The size of the image in m is given as x_width and y_width
    // Anything outside of this gets returned as 0

    // Our first implementation simply returns the floor value.
    assert(im_array.rows() == im_array.cols());
    // Size parameter of image
    int dimension = im_array.rows();
    double i_out=0;
    if ((abs(xt) < x_width) && (abs(yt) < y_width))
    {

        // Translate our position into the image co-ords
        int x_ID = int(xt/x_width*(dimension-1)/2 + (dimension-1)/2);
        int y_ID = int(yt/y_width*(dimension-1)/2 + (dimension-1)/2);
        i_out = im_array(x_ID, y_ID);
    } else {
        i_out=0.001;
    }
    return i_out;
}

double i_get_interp(MatrixXd &im_array, double xt, double yt,
                   double x_width, double y_width)
{
    double i_out=0;
    assert(im_array.rows() == im_array.cols());
    // Size parameter of image
    int dimension = im_array.rows();
    if ((abs(xt) < x_width) && (abs(yt) < y_width))
    {
        double idx = (xt/x_width*(dimension-1)/2 + (dimension-1)/2);
        double idy = (yt/y_width*(dimension-1)/2 + (dimension-1)/2);
        int x1 = int(idx);
        int y1 = int(idy);
        int y2 = y1+1;
        int x2 = x1+1;   

        // A couple of assertions to make sure we're in the right region
        assert(x1 < dimension);
        assert(y1 < dimension);
        assert(x2 < dimension);
        assert(y2 < dimension);
        assert(x1 >= 0);
        assert(y1 >= 0);
        assert(x2 >= 0);
        assert(y2 >= 0);
        
        double y1_interp = (x2-idx)*im_array(x1,y1)
                        + (idx-x1)*im_array(x2,y1);
        double y2_interp = (x2-idx)*im_array(x1,y2);
                        + (idx-x1)*im_array(x2,y2); 
        // Generate the fully interpolated value
        i_out = (y2-idy)*y1_interp + (idy-y1)*y2_interp;
        // assert((i_out > 0));
    } else {
        // Keep the output slightly above 0 to avoid some specific filtering in
        // My python code.
        i_out=0.001;
    }
    return i_out;
}

double i_get_kernel_old(MatrixXd &im_array, double xt, double yt,
                    double x_width, double y_width, MatrixXd &kernel,
                    double radius){
    // Gets the intensity by kernel multiplication -OLD VERSION WITH NO OPTIMISATION

    // This version is still the best coz no errors.
    assert(im_array.rows() == im_array.cols());
    int dimension = im_array.rows();
    int xk_width = kernel.rows();
    int yk_width = kernel.cols();
    double i_out = 0;

    for(int i=0; i<xk_width; i++){
        double x_temp = (i+0.5-xk_width/2)*(2*radius/xk_width)+xt;
        for(int j=0; j<xk_width; j++){
            // Convert kernel index value into x and y positions
            double y_temp = (j+0.5-yk_width/2)*(2*radius/yk_width)+yt;
            i_out += i_get_interp(im_array, x_temp, y_temp, x_width, y_width);
        }
    }
    // Output is normalised against number of elements of kernel (so increasing precision
    // doesn't increase the intensity in magnitude.)
    return i_out/(xk_width*yk_width);
}

int main (int argc, char ** argv) {
    // Read in the configuration file
    assert(argc>1);
    unordered_map<string, string> config = config_parse(argv[1]);

    // Simulation inputs
    //const double dt = 1*pow(10,-9);            // Simulation Time Step, Seconds
    const double dt = stof(config["dt"])*pow(10,-9);
    const double dt_sample = stof(config["dt_sample"])*pow(10,-9);    // Wirteout timestep, seconds
    const int integration_steps = stoi(config["integration_steps"]);           // How many writeout timesteps per intensity integration
    const int samples_per_position = stoi(config["samples_per_position"]);   // How many integration samples per scan point
    const int N_dt_samples = integration_steps*samples_per_position;
    //cout<<dt<<' '<<dt_sample<<' '<<integration_steps<<' '<<N_dt_samples<<'\n';
    // What is the uncertainty in position measurement?
    const double position_noise_sigma = stof(config["position_noise_sigma"])*pow(10,-9);

    const int dts_per_sample = dt_sample/dt;
    string output_name = config["output_loc"];                     // Name of our output file
    cout << "Beginning Simulation: " << output_name << '\n';

    // Scanning Inputs
    const double x_scan_width = stof(config["x_scan_width"])*pow(10,-9);  // Width of scanning, we do +/- this from O (metres)
    const double y_scan_width = stof(config["y_scan_width"])*pow(10,-9);        
    const int x_scan_points = stoi(config["x_scan_points"]);               // How many points in X and Y will we use?
    const int y_scan_points = stoi(config["y_scan_points"]);
    /*
    const double x_scan_width = 750*pow(10,-9);  // Width of scanning, we do +/- this from O (metres)
    const double y_scan_width = 750*pow(10,-9);        
    const int x_scan_points = 100;               // How many points in X and Y will we use?
    const int y_scan_points = 100;
    */
    // Object Inputs
    const double radius = stof(config["radius"])*pow(10,-9);      // Metres   
    const double density = stof(config["object_density"]);     // kg/m^3 
    const double mass = 4/3*PI*pow(radius,3)*density; //kg
    cout << "Mass is " <<mass << "kg \n";
    cout << "Dts per sample " << dts_per_sample << " \n";
    /*
    Density will need to be looked up for each material.
    Density of polysytrene spheres is 1.05 g/cm^3, taken from
    http://www.polysciences.com/skin/frontend/default/polysciences/pdf/TDS%20238.pdf
    Density of diamond is taken as 3 g/cm^3, taken from
    https://www.engineeringtoolbox.com/density-solids-d_1265.html
    */

    // Fluid / trap inputs
    const int temp = stoi(config["temp"]);                      // Kelvin
    const double dyn_visc = stof(config["dyn_visc"]);              // N s/ m^2
    //const double stiffness_x = 4.62*pow(10,-5);    // N/m
    //const double stiffness_y = 8.25*pow(10,-6);    //N/m
    const double stiffness_x = stof(config["stiffness_x"]);    // N/m
    const double stiffness_y = stof(config["stiffness_y"]);    //N/m
    //Stiffnesses are taken from May 16 experimental data
    //const double stiffness = 0;
    //Cycle between 0 stiffness and some value, to test either diffusion or MSD/Stiffness
    /*
    dyn_visc Taken from https://www.engineeringtoolbox.com/
    water-dynamic-kinematic-viscosity-d_596.html
    */

    // Intensity Image Inputs
    string img_loc = config["img_loc"];
    //int dimension = 1200;
    int dimension = stoi(config["dimension"]);       // The length of each side of the 2D image array
    MatrixXd im_array; // Create a 2D array of doubles
    im_array.resize(dimension, dimension); // Resize array to correct dimensions
    double x_width = stof(config["x_width"])*pow(10,-6);       // The width of the image to be scaled, metres
    double y_width = stof(config["x_width"])*pow(10,-6);

    // Image convolution Kernel
    int kernel_width = stoi(config["kernel_width"]);
    MatrixXd kernel;
    if (kernel_width != 0){
        // Generate our kernel (so far I am assuming a circle.)
        kernel.resize(kernel_width*2, kernel_width*2); // Resize array to correct dimension
        for(int i=0; i<kernel_width*2; i++){
            for(int j=0; j<kernel_width*2; j++){
                if(sqrt(pow((i-kernel_width),2)+pow((j-kernel_width),2)) <= kernel_width){
                    kernel(i,j) = 1;
                } else {
                    kernel(i,j) = 0;
                }
            }
        }
    } else {
        // We're just having a point source!
        kernel.resize(1,1);
        kernel(0,0) = 1;
    } 
    cout<<"The Kernel Looks like This:"<<'\n';
    print_array(kernel);

    // Other Constants
    // We need to create the random number generator outside of the function call!
    unsigned seed = chrono::system_clock::now().time_since_epoch().count();
    default_random_engine generator (seed);
    // A second random that's not related to the first.
    unsigned seed2 = chrono::system_clock::now().time_since_epoch().count()+5;
    default_random_engine generator2 (seed);
    // A normal distribution to pull from - used in brownianing-around the particle
    normal_distribution<double> distribution(0.0,1.0);
    // A noisy distribution to add noise to our position outputs!
    normal_distribution<double> noise_boi(0.0, position_noise_sigma);
    const double k_b = 1.38*pow(10,-23);      // Boltzmann's constant N m / K

    // Calculated parameters
    double drag_coeff = 6*PI*dyn_visc*radius; 
    double diffusion_constant = k_b*temp/drag_coeff;
    double chi = sqrt(2*drag_coeff*k_b*temp/dt);
    cout << "Diffusion constant is " <<diffusion_constant << "m^2/s \n";
    cout << "Chi constant is " <<chi << "N \n";
    cout << "Drag Coeff is " << drag_coeff << '\n';
    // Initial Conditions 
    size_t size = (N_dt_samples*x_scan_points*y_scan_points);
    double xt = 0;
    double yt = 0;
    double axt = 0;
    double ayt = 0;
    double vxt = 0;
    double vyt = 0;
    double i_int=0;
    int a = 0;
    // I am using vectors (instead of the eigen equivalent) for variables that don't
    // need to have complex vector operations.
    vector<double> t_out(size);
    vector<double> x_out(size);
    vector<double> y_out(size);
    vector<double> i_out(size);
    vector<double> x_centre_out(size);
    vector<double> y_centre_out(size);
    vector<int> write_index(size);
    int samples_written = 0;
}

