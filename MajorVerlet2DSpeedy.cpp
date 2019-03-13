/*
This is a program that should simulate output of my NSM scanner,

It keeps a continuously diffusing / trapped sphere and varies a static offset
(which is the scanning parameter), and uses the combined position to infer 
an intensity from a given image. Output is combined position, time, and intensities

Compiler Command: g++ -O3 MajorVerlet2DSpeedy.cpp -o Major
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

using namespace std;

#define PI 3.14159265359 

void print_array(vector<vector<double>> &im_array){
    // A function that prints the array to screen.
    for(int i=0; i<im_array.size(); i++){
        for(int j=0; j<im_array[0].size(); j++){
            cout<<im_array[i][j]<<' ';
        }
        cout<<'\n';
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

double i_get(vector<vector<double>> &im_array, double xt, double yt,
            double x_width, double y_width, int dimension)
{
    // This function returns the intensity at point rt
    // Intensity is given by 2D vector array img_array
    // The size of the image in m is given as x_width and y_width
    // Anything outside of this gets returned as 0

    // Our first implementation simply returns the floor value.
    double i_out=0;
    int x_ID=0;
    int y_ID=0;
    if ((abs(xt) < x_width) && (abs(yt) < y_width))
    {

        // Translate our position into the image co-ords
        x_ID = int(xt/x_width*(dimension-1)/2 + (dimension-1)/2);
        y_ID = int(yt/y_width*(dimension-1)/2 + (dimension-1)/2);
        i_out = im_array[x_ID][y_ID];
    } else {
        i_out=0.001;
    }


    return i_out;
}

double i_get_interp(vector<vector<double>> &im_array, double xt, double yt,
                   double x_width, double y_width, int dimension)
{
    double i_out=0;
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
        
        double y1_interp = (x2-idx)*im_array[x1][y1]
                        + (idx-x1)*im_array[x2][y1];
        double y2_interp = (x2-idx)*im_array[x1][y2]
                        + (idx-x1)*im_array[x2][y2]; 
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


double i_get_kernel(vector<vector<double>> &im_array, double xt, double yt,
                    double x_width, double y_width, vector<vector<double>> &kernel,
                    double radius){
    // Gets the intensity by kernel multiplication.
    double i_out=0.001;
    int dimension = im_array.size();
    int xk_width = kernel.size();
    int yk_width = kernel[0].size();

    double xk_pos[xk_width];
    double yk_pos[yk_width];

    double idx;
    double idy;
    int x1;
    int y1;
    int y2;
    int x2;  

    //Pre-compute kernel array:
    for(int i=0; i<xk_width; i++){
        xk_pos[i]= (i+0.5-xk_width/2)*(2*radius/xk_width)+xt;
    }
    for(int j=0; j<yk_width; j++){
        // Convert kernel index value into x and y positions
        yk_pos[j] = (j+0.5-yk_width/2)*(2*radius/yk_width)+yt;
    }
    for(int i=0; i<xk_width; i++){
        for(int j=0; j<yk_width; j++){
            //i_out += i_get_interp(im_array, xk_pos[i], yk_pos[j], x_width, y_width, dimension);
            idx = xk_pos[i]/x_width*(dimension-1)/2 + (dimension-1)/2;
            idy = yk_pos[j]/y_width*(dimension-1)/2 + (dimension-1)/2;
            x1 = max(0, min(int(idx), dimension-2));
            y1 = max(0, min(int(idy), dimension-2));
            y2 = y1+1;
            x2 = x1+1;   

            // A couple of assertions to make sure we're in the right region
            assert(x2 < dimension);
            assert(y2 < dimension);
            assert(x1 >= 0);
            assert(y1 >= 0);

            // Generate the intensity
            double y1_interp = (x2-idx)*im_array[x1][y1]
                            + (idx-x1)*im_array[x2][y1];
            double y2_interp = (x2-idx)*im_array[x1][y2]
                            + (idx-x1)*im_array[x2][y2]; 
                            
            i_out += (y2-idy)*y1_interp + (idy-y1)*y2_interp;
        }
    }
    // Output is normalised against number of elements of kernel (so increasing precision
    // doesn't increase the intensity in magnitude.)
    return max(double(0),i_out/(xk_width*yk_width));
}

double i_get_kernel_old(vector<vector<double>> &im_array, double xt, double yt,
                    double x_width, double y_width, vector<vector<double>> &kernel,
                    double radius){
    // Gets the intensity by kernel multiplication -OLD VERSION WITH NO OPTIMISATION
    // This version is still the best coz no errors.
    double i_out=0;
    int x_ID=0;
    int y_ID=0;
    int dimension = im_array.size();

    int xk_width = kernel.size();
    int yk_width = kernel[0].size();

    double x_temp = 0;
    double y_temp = 0;

    for(int i=0; i<xk_width; i++){
        x_temp = (i+0.5-xk_width/2)*(2*radius/xk_width)+xt;
        for(int j=0; j<yk_width; j++){
            // Convert kernel index value into x and y positions
            y_temp = (j+0.5-yk_width/2)*(2*radius/yk_width)+yt;
            i_out += i_get_interp(im_array, x_temp, y_temp, x_width, y_width, dimension);
        }
    }
    // Output is normalised against number of elements of kernel (so increasing precision
    // doesn't increase the intensity in magnitude.)
    return i_out/(xk_width*yk_width);
}

int main (int argc, char ** argv) 
{
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
    vector<vector<double>> im_array; // Create a 2D array of doubles
    im_array.resize(dimension, vector<double>(dimension,0)); // Resize array to correct dimensions
    double x_width = stof(config["x_width"])*pow(10,-6);       // The width of the image to be scaled, metres
    double y_width = stof(config["x_width"])*pow(10,-6);

    // Image convolution Kernel
    int kernel_width = stoi(config["kernel_width"]);
    vector<vector<double>> kernel;
    if (kernel_width != 0){
        // Generate our kernel (so far I am assuming a circle.)
        kernel.resize(kernel_width*2, vector<double>(kernel_width*2,0)); // Resize array to correct dimension
        for(int i=0; i<kernel_width*2; i++){
            for(int j=0; j<kernel_width*2; j++){
                if(sqrt(pow((i-kernel_width),2)+pow((j-kernel_width),2)) <= kernel_width){
                    kernel[i][j] = 1;
                } else {
                    kernel[i][j] = 0;
                }
            }
        }
    } else {
        // We're just having a point source!
        kernel.resize(1, vector<double>(1,0));
        kernel[0][0] = 1;
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
    vector<double> t_out(size);
    vector<double> x_out(size);
    vector<double> y_out(size);
    vector<double> i_out(size);
    vector<double> x_centre_out(size);
    vector<double> y_centre_out(size);
    vector<int> write_index(size);
    int samples_written = 0;

    // Create array of intensities from a saved image
    cout<<"Reading Image "<<img_loc<<'\n';
    ifstream file(img_loc);
    for (int row=0; row < dimension-1; ++row)
    {
        string line;            // Create a new vairable called line
        getline(file,line);     // Get one row and put it in 'line'
        if (!file.good())       // If the file is shit
        {
            cout<<"BAD FILE!\n";
            exit(5);
        }
        stringstream iss(line);  // Turn 'line' into a stringstream
        for(int col = 0; col < dimension-1; ++ col)
        {
            string val;
            getline(iss, val, ','); // Get one value and put it in val        
            if (!iss.good())
            {
                cout<<"BAD VALUE!\n";
                break;
            }
            stringstream convertor(val);        
            convertor >> im_array[row][col];    // Use the extract action into im_array
        }
    }

    // Generating a 2x2 vector of X and Y trap centre positions
    cout<<"Generating Trap Centres\n";
    vector<double> x_trap_centres;
    vector<double> y_trap_centres;
    x_trap_centres.resize(x_scan_points,0); // Resize array to correct dimensions
    y_trap_centres.resize(y_scan_points,0);
    for(int row = 0; row < x_scan_points; ++row)
    {
       x_trap_centres[row] = (row - x_scan_points/2)*(2*x_scan_width/x_scan_points);
    }
    for(int row = 0; row < y_scan_points; ++row)
    {
       y_trap_centres[row] = (row - y_scan_points/2)*(2*y_scan_width/y_scan_points);
    }


    // Copy the config file along with the results
    ifstream src(argv[1], ios::binary);
    ofstream dst(output_name+"_config.txt");
    dst << src.rdbuf();

    unsigned long long x_end = 0;
    unsigned long long y_end = 0;
    unsigned long long i_end = 0;
    unsigned long long write_end = 0;
    unsigned long long x_end_tot = 0;
    unsigned long long y_end_tot = 0;
    unsigned long long i_end_tot = 0;
    // Initialise 'trap position' values
    int x_trap_pos = 0;
    int y_trap_pos = 0;
    int write_dex = 0; // A writing index
    bool go = 1;

    // Start loop timing
    cout<<"Beginning loop timer\n";
    auto start = chrono::high_resolution_clock::now();
    //for(int yeet=0; yeet<100; ++yeet){
    auto l_start = chrono::high_resolution_clock::now();
    while(go){  
        double vxt_dt=0;
        double vyt_dt=0;
        double f_out=0;
        double xt_dt=0;
        double yt_dt=0;
        double axt_dt=0;
        double ayt_dt=0;
        double randx_term=0;
        double randy_term=0;
        double i_dt=0;
        //X-Dimension
        auto x_start = chrono::high_resolution_clock::now();
        randx_term = distribution(generator);
        f_out = f_get(stiffness_x,xt);
        axt_dt = a_step(vxt,drag_coeff,mass,chi,f_out,randx_term);
        vxt_dt = v_step(vxt,axt,axt_dt,dt);
        xt_dt = r_step(xt,vxt,axt,dt);

        xt = xt_dt;
        vxt = vxt_dt;
        axt = axt_dt;
            // Output some timings for the X dimensiopn
        auto temp = chrono::duration_cast<chrono::nanoseconds>(chrono::high_resolution_clock::now() - x_start).count();
        x_end += temp;
        x_end_tot += temp;

        //Y-Dimension
        auto y_start = chrono::high_resolution_clock::now();
        randy_term = distribution(generator2);
        f_out = f_get(stiffness_y,yt);
        ayt_dt = a_step(vyt,drag_coeff,mass,chi,f_out,randy_term);
        vyt_dt = v_step(vyt,ayt,ayt_dt,dt);
        yt_dt = r_step(yt,vyt,ayt,dt);

        yt = yt_dt;
        vyt = vyt_dt;
        ayt = ayt_dt;
            // Output some timings for the Y dimension
        temp = chrono::duration_cast<chrono::nanoseconds>(chrono::high_resolution_clock::now() - y_start).count();
        y_end += temp;
        y_end_tot += temp;


        //Get the intensity (of current position) and add it to our integration
        //We add our current brownian position to the trap centre
        auto i_start = chrono::high_resolution_clock::now();
        /*
        i_dt = i_get_interp(im_array,
            xt+x_trap_centres[x_trap_pos],yt+y_trap_centres[y_trap_pos],
            x_width,y_width,dimension);
        */
        i_dt = i_get_kernel_old(im_array,
            xt+x_trap_centres[x_trap_pos],yt+y_trap_centres[y_trap_pos],
            x_width, y_width, kernel, radius);
        i_int = i_int + i_dt;
        
        temp = chrono::duration_cast<chrono::nanoseconds>(chrono::high_resolution_clock::now() - i_start).count();
        i_end += temp;
        i_end_tot += temp;

        // Writouts segment
        auto write_start = chrono::high_resolution_clock::now();
        if (a%dts_per_sample == 0)
        {
            // Write out position details! We are adding NOISE to the position. 
            x_out[samples_written] = xt+x_trap_centres[x_trap_pos]+noise_boi(generator);
            y_out[samples_written] = yt+y_trap_centres[y_trap_pos]+noise_boi(generator2);
            t_out[samples_written] = samples_written*dt_sample;
            x_centre_out[samples_written] = x_trap_centres[x_trap_pos];            
            y_centre_out[samples_written] = y_trap_centres[y_trap_pos];
            if (samples_written%integration_steps == 0)
            {
                //We are in an intensity writeout step
                i_out[samples_written] = i_int;
                //cout<<i_int<<'\n';
                i_int = 0;
                write_dex = 0;
                write_index[samples_written] = write_dex;
                if ((y_trap_pos == y_scan_points-1) and (x_trap_pos == x_scan_points-1)){
                    // Get rid of a fenceposting error on the very last scan point
                    i_out[samples_written] = 0;
                }
            } else {
                //Write zero to the intensity out
                i_out[samples_written] = 0;
                write_dex++;
                write_index[samples_written] = write_dex;
            }
            samples_written++;
            // Handle our moving of trap centre, which occurs every
            // N_dt_samples samples
            if (samples_written%N_dt_samples==0)
            {
                // We have written out enough times for this specic trap position
                y_trap_pos++;
                if (y_trap_pos > y_scan_points-1)
                {
                    // Jump along x, reset y
                    y_trap_pos = 0;
                    x_trap_pos++;
                    cout<<"Moving to row "<<x_trap_pos+1<<" of "<<x_scan_points<<'\n';
                    cout<<"This step took "<< chrono::duration_cast<chrono::microseconds>(chrono::high_resolution_clock::now() - l_start).count()/(double)1000000
                    <<" seconds\n";
                    l_start = chrono::high_resolution_clock::now();
                    cout << "X-position Loop Average Time (ns): " << x_end/(double)a<<'\n';
                    cout << "Y-position Loop Average Time (ns): " << y_end/(double)a <<'\n';
                    cout << "Intensity Loop Average Time (ns): " << i_end/(double)a <<'\n';
                    x_end = 0;
                    y_end = 0;
                    i_end = 0;
                    a=0; // Reset 'a' to prevent overflow - we don't need an absolute count
                }
                if (x_trap_pos > x_scan_points-1)
                {
                    // Exit our loop if we reach the end
                    go = 0;
                }

            }
        }
        a++;
        write_end += chrono::duration_cast<chrono::nanoseconds>(chrono::high_resolution_clock::now() - write_start).count();
    }
    auto elapsed = chrono::high_resolution_clock::now() - start;
    long long microseconds = chrono::duration_cast<chrono::microseconds>(elapsed).count();

    // Write stuff out (We have some separate outfiles for specific analysis.)
    ofstream info_out_file(output_name+"_info_out.txt");
    ofstream kernel_out(output_name+"_kernel.txt");
    ofstream combo_out_file(output_name+"_combo_out.txt");
    ofstream timing_out_file(output_name+"_timing_out.txt");
    // Write everything to .txt
    for (int i = 1; i < size; ++i){
        combo_out_file << t_out[i] << '\t'
        << x_out[i] << '\t'
        << y_out[i] << '\t'
        << i_out[i] << '\t'
        << x_centre_out[i] << '\t'
        << y_centre_out[i] << '\t'
        << write_index[i] << '\n';
    }
    // Write our info file
    info_out_file << dt << '\t'     // Simulation timestep
    << dt_sample << '\t'            // Writeout timestep
    << integration_steps <<'\t'     // How many writeouts per intensity integration?
    << N_dt_samples << '\t'         // How many samples per point
    << stiffness_x << '\t' << stiffness_y << '\t'
    << x_scan_width << '\t' << y_scan_width <<'\t'
    << x_scan_points << '\t' << y_scan_points << '\t'
    << img_loc << '\t' << x_width << '\t' <<y_width << '\t';

    // Write out the kernel:
    for(int i=0; i<kernel.size(); i++){
        for(int j=0; j<kernel[0].size(); j++){
            kernel_out<<kernel[i][j]<<'\t';
        }
        kernel_out<<'\n';
    } 

    timing_out_file << "\nTIMING DETAILS \nTotal time in seconds is " << microseconds/1000000 <<'\n'
    << "With a total of " << size 
    << " outputs, this is " << microseconds/size
    << " us per output\n"
    << "With a timestep size of " << dt << " s, and duration-of-output of "<< dt_sample << " s"
    << ",\nthis is a total of " 
    << (double)microseconds/(double)size/(double)dts_per_sample
    << " microseconds per single-loop.\n"
    << "X-position Loop Average Time (ns): " << x_end_tot/(double)size/(double)dts_per_sample <<'\n'
    << "Y-position Loop Average Time (ns): " << y_end_tot/(double)size/(double)dts_per_sample <<'\n'
    << "Intensity Loop Average Time (ns): " << i_end_tot/(double)size/(double)dts_per_sample <<'\n'
    << "Write-Loop Average Time (ns): "<< write_end/(double)size/(double)dts_per_sample <<'\n'
    << "Estimated total-time taken [SHOULD MATCH THE ABOVE] (s): " << (x_end_tot+y_end_tot+i_end_tot+write_end)/double(1000000000) <<'\n';

    cout << "Running Complete! Look at timing log file for analysis.\n";


    return 0;
}