//
//  main.cpp
//  DDAv1
//
//  Created by Payton Beeler on 1/9/19.
//  Copyright © 2019 Payton Beeler. All rights reserved.
//

#include <ctime>
#include <iostream>
#include <math.h>
#include <iomanip>
#include <vector>
#include <cmath>
#include <fstream>
#include <cstdio>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <float.h>

using namespace std;

#define file_format 3 //1=mn,x,y,z 2=x,y,z, 3=x,y,z,r with header
#define radius_monomer 0.020 //radius of the monomer in micron
#define dummy_radius 0.5 //radius of the monomers in dummy units
#define mode 2 // mode 0 = layers; mode 1 = mass percent; 2 = in junctions
#define Rd 6    //dipoles per radius, keep to at least 12 dipoles per 1.0 dummy radius
#define seed time(NULL)     //random number generator seed
#define PI 4*atan(1)

vector<vector<double> >build_vector ();
vector<vector<double> >recenter(vector<vector<double> >, double);
int max(vector<vector<double> >);
double length(vector<vector<double> >, int);
vector<vector<double> > buffer(vector<vector<double> >, double, int);
void find_core(vector<vector<double> >, double, int, int **shape_file, double **cartesian_space, int **monomer_centers);
double AggregateMass(int **shape_file, int, double, double);
double AggregateVolume(int **shape_file, double, int);
double CoreVolume(int **shape_file, double, int);
void neighbors(int **shape_file, int, int *neighbor);
void coat_even(int **shape_file, int *neighbor, int, double, double, int **shape_copy);
void coat_junctions(int **shape_file, int** monomer_centers, int, double, double, int **shape_copy, int, double);
double psdrand(int);
vector<double>area(int **shape_file, int, int **shape_copy);
double SA_fraction(int **shape_file, int);
vector<double>center_mass (vector<vector<double> >v);



int main()
{
    const double rho_core = 1.8;  //density of the core
    const double rho_coating = 1.2; //density of the coating
    
    time_t start1, end1,start2, end2;
    time(&start1);

    srand(seed);
    int **shape_file, **monomer_centers, max_size, *neighbor, **shape_copy, minimum, maximum, monomers, buffer_layers;
    double layers=0, dipole_length, dipole_center, **cartesian_space, mass_core, volume, bare_eq_radius, coated_eq_radius, mass_coating, mass_p, sum, average, core_volume, diameter=2*double(dummy_radius), volume_core_2, coating_thickness, F, max_bare_area, max_coated_area;
    vector<vector<double> >original;
    vector<double>cross_section;
    vector<double>average_area(3);
    double Vratio, mtot_mbc, P_cs, P_cc, mass_percent;
    
    if (mode==1 || mode == 2)
    {
        cout << "P coating-surface? ";
        cin >> P_cs;
        cout << "P coating-coating? ";
        cin >> P_cc;
        cout << "V/V0? ";
        cin >> Vratio;
        buffer_layers=8*Vratio+5;
        mtot_mbc = 1+(rho_coating/rho_core)*(Vratio-1);
        mass_percent = mtot_mbc-1;
        cout << "Mtot/Mbc: " << mtot_mbc << endl;
    }
    
    else
    {
        cout << "Layers to add: " << coating_thickness << endl;
        cout << endl;
    }
 
    
    cout << "Building vector...";
    original=build_vector();
    monomers=int(original.size());
    original=recenter(original, diameter);
    cout << "finished" << endl;
    cout << endl;
    cout << "Finding core...";
    max_size=max(original);
    dipole_length=length(original, max_size);
    cout << endl;
    cout << "dipole length = " << dipole_length << endl;
    dipole_center=dipole_length/2;
    original=buffer(original, dipole_length, buffer_layers);
    if (mode==0)
    {
        max_size=max_size+(2*buffer_layers)+(2*coating_thickness);
    }
    else
    {
        max_size=max_size+2*buffer_layers;
    }
    
    cout << "number of dipoles in lattice = " << max_size << endl;
    
    //  MEMORY ALLOCATION
    //  -----------------------------------------------------------------------------------//
    shape_file=(int **)malloc(max_size*max_size*max_size*sizeof (int *));                //
    if(shape_file==NULL)                                                               //
    {                                                                                  //
        printf("Shape file memory error");                                             //
    }                                                                                  //
    else
    {
        cout << "Shape file memory allocated" << endl;
    }
    for(int i=0; i<max_size*max_size*max_size; i++)                                    //
    {                                                                                  //
        shape_file[i]=(int *)malloc(4*sizeof(int));                                     //
        if (shape_file[i]==NULL)                                                       //
        {                                                                              //
            printf("Shape file memory error");                                         //
        }                                                                              //
    }                                                                                  //
    
    shape_copy=(int **)malloc(max_size*max_size*max_size*sizeof (int *));                //
    if(shape_copy==NULL)                                                               //
    {                                                                                  //
        printf("Shape copy file memory error");                                        //
    }                                                                                  //
    else
    {
        cout << "Shape copy memory allocated" << endl;
    }
    for(int i=0; i<max_size*max_size*max_size; i++)                                    //
    {                                                                                  //
        shape_copy[i]=(int *)malloc(4*sizeof(int));                                     //
        if (shape_copy[i]==NULL)                                                       //
        {                                                                              //
            printf("Shape copy file memory error");                                    //
        }                                                                              //
    }                                                                                  //
    //
    monomer_centers=(int **)malloc(monomers*monomers*monomers*sizeof (int *));         //
    if(monomer_centers==NULL)                                                          //
    {                                                                                  //
        printf("Monomer center file memory error");                               //
    }                                                                                  //
    else
    {
        cout << "Monomer center memory allocated" << endl;
    }
    for(int i=0; i<monomers*monomers*monomers; i++)                                    //
    {                                                                                  //
        monomer_centers[i]=(int *)malloc(3*sizeof(int));                               //
        if (monomer_centers[i]==NULL)                                                  //
        {                                                                              //
            printf("Monomer center file memory error");                                //
        }                                                                              //
    }
    cartesian_space=(double **)malloc(max_size*max_size*max_size*sizeof (double *));   //
    if(cartesian_space==NULL)                                                          //
    {                                                                                  //
        printf("Cartesian memory error");                                              //
    }                                                                                  //
    else
    {
        cout << "Cartesian memory allocated" << endl;
    }
    for(int i=0; i<max_size*max_size*max_size; i++)                                    //
    {                                                                                  //
        cartesian_space[i]=(double *)malloc(3*sizeof(double));                          //
        if (cartesian_space[i]==NULL)                                                  //
        {                                                                              //
            printf("Cartesian file memory error");                                     //
        }                                                                              //
    }                                                                                  //
    
    //
    neighbor=(int *)malloc(max_size*max_size*max_size*sizeof(int));                     //
    if(neighbor==NULL)                                                                 //
    {                                                                                  //
        printf("Neighbor memory error");                                               //
    }                                                                                  //
    else
    {
        cout << "Neighbor memory allocated" << endl;
    }
    //-------------------------------------------------------------------------------------//
    
    
    if (mode==0)
    {
        find_core(original, dipole_center, max_size, shape_file, cartesian_space, monomer_centers);
        cout << "finished" << endl;
        
        volume=AggregateVolume(shape_file, dipole_length, max_size);
        volume=(volume*radius_monomer*radius_monomer*radius_monomer)/(dummy_radius*dummy_radius*dummy_radius);
        volume_core_2=volume;
        mass_core = rho_core*(volume);
        
        volume=AggregateVolume(shape_file, dipole_length, max_size);
        core_volume=volume;
        bare_eq_radius=cbrt((3*volume)/(4*3.141592654));
        
        cout << endl;
        cout << "Layers applied: " << endl;
        do
        {
            neighbors(shape_file, max_size, neighbor);
            coat_even(shape_file, neighbor, max_size, P_cs, P_cc, shape_copy);
            volume=AggregateVolume(shape_file, dipole_length, max_size);
            coated_eq_radius=cbrt((3*volume)/(4*3.141592654));
            layers = (coated_eq_radius-bare_eq_radius)/dipole_length;
            cout << layers << endl;
            
        } while (layers<coating_thickness);
        
        volume=AggregateVolume(shape_file, dipole_length, max_size);
        volume=(volume*radius_monomer*radius_monomer*radius_monomer)/(dummy_radius*dummy_radius*dummy_radius);
        mass_coating = rho_coating*(volume-volume_core_2);
        
    }
    
    else if (mode==1)
    {        
        find_core(original, dipole_center, max_size, shape_file, cartesian_space, monomer_centers);
        cout << "finished" << endl;
        
        time(&end1);
        time(&start2);
        
        volume=AggregateVolume(shape_file, dipole_length, max_size);
        volume=(volume*radius_monomer*radius_monomer*radius_monomer)/(dummy_radius*dummy_radius*dummy_radius);
        core_volume = volume;
        bare_eq_radius=cbrt((3*volume)/(4*PI));
        
        cout << endl;
        cout << "Adding mass percent... " << endl;
        do
        {
            neighbors(shape_file, max_size, neighbor);
            if (mtot_mbc>1)
            {
                coat_even(shape_file, neighbor, max_size, P_cs, P_cc, shape_copy);
            }
            volume=AggregateVolume(shape_file, dipole_length, max_size);
            volume=(volume*radius_monomer*radius_monomer*radius_monomer)/(dummy_radius*dummy_radius*dummy_radius);
            coated_eq_radius=cbrt((3*volume)/(4*PI));
            mass_core = rho_core*(core_volume);
            mass_coating = rho_coating*(volume-core_volume);
            mass_p = mass_coating/mass_core;
            cout << mass_p << endl;
            if (mtot_mbc>1 && mass_p==0)
            {
               exit(EXIT_FAILURE);
            }
            
        } while (mass_p<mass_percent);
        
        F=SA_fraction(shape_file, max_size);
        
        cout << "finished" << endl;
    }
    else if (mode==2)
    {
        find_core(original, dipole_center, max_size, shape_file, cartesian_space, monomer_centers);
        cout << "finished" << endl;
        
        time(&end1);
        time(&start2);
        
        volume=AggregateVolume(shape_file, dipole_length, max_size);
        volume=(volume*radius_monomer*radius_monomer*radius_monomer)/(dummy_radius*dummy_radius*dummy_radius);
        core_volume = volume;
        bare_eq_radius=cbrt((3*volume)/(4*PI));
        coated_eq_radius=bare_eq_radius;
        
        cout << endl;
        cout << "Adding mass via monomer junctions... " << endl;
        
        // add a little coating to the junctions
        neighbors(shape_file, max_size, neighbor);
        double gamma;
        if (mtot_mbc>1)
        {
            for (int i=0; i<10; i++)
            {
                gamma = double(i+1)/10;
                coat_junctions(shape_file, monomer_centers, max_size, P_cs, P_cc, shape_copy, monomers, gamma);
                volume=AggregateVolume(shape_file, dipole_length, max_size);
                volume=(volume*radius_monomer*radius_monomer*radius_monomer)/(dummy_radius*dummy_radius*dummy_radius);
                coated_eq_radius=cbrt((3*volume)/(4*PI));
                mass_core = rho_core*(core_volume);
                mass_coating = rho_coating*(volume-core_volume);
                mass_p = mass_coating/mass_core;
                cout << mass_p << endl;
                if (mass_p>=mass_percent)
                {
                    break;
                }
            }
            if (mass_p<mass_percent)
            {
                do
                {
                    neighbors(shape_file, max_size, neighbor);
                    coat_even(shape_file, neighbor, max_size, P_cs, P_cc, shape_copy);
                    volume=AggregateVolume(shape_file, dipole_length, max_size);
                    volume=(volume*radius_monomer*radius_monomer*radius_monomer)/(dummy_radius*dummy_radius*dummy_radius);
                    coated_eq_radius=cbrt((3*volume)/(4*PI));
                    mass_core = rho_core*(core_volume);
                    mass_coating = rho_coating*(volume-core_volume);
                    mass_p = mass_coating/mass_core;
                    cout << mass_p << endl;
                    if (mtot_mbc>1 && mass_p==0)
                    {
                        exit(EXIT_FAILURE);
                    }
                    
                } while (mass_p<mass_percent);
            }
            F=SA_fraction(shape_file, max_size);
            cout << "finished" << endl;
        }
    }
    
    
    else
    {
        cout << "Mode error" << endl;
        exit(EXIT_FAILURE);
    }
    
    
    cout << endl;
    cout << "Finding cross sections...";
    
    cross_section=area(shape_file, max_size, shape_copy);
    
    max_bare_area = 0;
    for (int i=0; i<3; i++)
    {
        if (cross_section[i]>max_bare_area)
        {
            max_bare_area=cross_section[i];
        }
    }
    
    max_bare_area=max_bare_area*dipole_length*dipole_length;
    max_bare_area=(max_bare_area*radius_monomer*radius_monomer)/(dummy_radius*dummy_radius);
    
    max_coated_area = 0;
    for (int i=3; i<cross_section.size(); i++)
    {
        if (cross_section[i]>max_coated_area)
        {
            max_coated_area=cross_section[i];
        }
    }
    
    max_coated_area=max_coated_area*dipole_length*dipole_length;
    max_coated_area=(max_coated_area*radius_monomer*radius_monomer)/(dummy_radius*dummy_radius);
    cout << "finished" << endl;
    cout << endl;
    cout << "Writing...";

    std::ofstream outfile2 ("coated_particle");
    if (outfile2.is_open())
    {
        outfile2 << "Nmat=2" << endl;

        for (int i=0; i<max_size*max_size*max_size; i++)
        {
            if (shape_file[i][3]!=0)
            {
                outfile2 << shape_file[i][0] << " " << shape_file[i][1] << " " << shape_file[i][2] << " " << shape_file[i][3] << endl;
            }
        }
        outfile2.close();
    }
    
    ofstream areafile ("Cluster_Data.txt");  //this will give the aggregate parameters
    if (areafile.is_open())
    {
        areafile << "Number of monomers: " << monomers << endl;
        areafile << "Radius of monomers: " << radius_monomer << " micron" << endl;
        areafile << "Core mobility diameter: " << 2.0*sqrt((1/(4.0*atan(1)))*max_bare_area) << " micron" << endl;
        areafile << "Coated mobility diameter: " << 2.0*sqrt((1/(4.0*atan(1)))*max_coated_area) << " micron" << endl;
        areafile << "Bare equivalent radius: " << bare_eq_radius << " micron" << endl;
        areafile << "Coated equivalent radius: " << coated_eq_radius << " micron" << endl;
        areafile << "Coated surface area fraction: " << F << endl;
        if (mode==0)
        {
            areafile << "Initial mass: " << mass_core*1E-12 << " g" << endl;
            areafile << "Final mass: " << (mass_core+mass_coating)*1E-12 << " g" << endl;
            areafile << "Layers of coating: " << layers << endl;
            areafile << "Mtot/Mbc: " << (mass_core+mass_coating)/mass_core << endl;
        }
        else if (mode==1 || mode==2)
        {
            areafile << "Initial mass: " << mass_core*1E-12 << " g" << endl;
            areafile << "Final mass: " << (mass_core+mass_coating)*1E-12 << " g" << endl;
            areafile << "Mass percent: " << mass_p*100 << "%" << endl;
            areafile << "Mtot/Mbc: " << mass_p+1 << endl;
        }


        areafile.close();
    }
    cout << "finished" << endl;

    time(&end2);
    double time_taken1 = double(end1 - start1);
    double time_taken2 = double(end2 - start2);
    double total_time = double(end2 - start1);
    cout << "Time for core : " << fixed << time_taken1 << setprecision(5) << " seconds " << endl;
    cout << "Time for coating : " << fixed << time_taken2 << setprecision(5) << " seconds " << endl;
    cout << "Total time : " << fixed << total_time << setprecision(5) << " seconds " << endl;
    cout << endl;
    
}

//-----------------------------------read csv file and turn it into a vector-------------------------------
vector<vector<double> >build_vector ()
{
    vector<vector<double> >v;
    int size, row;
    vector<double>temp(4);
    ifstream test;
    test.open("particle");
    
    if (!test.is_open())
    {
        cout <<"FILE ERROR" << endl;
        exit(EXIT_FAILURE);
    }
    
    double x, y, z, mn;
    
    

    if (file_format==1)
    {
        while(!test.eof())
        {
            test >> mn;
            test >> x;
            test >> y;
            test >> z;
                
            temp.at(0)=mn;
            temp.at(1)=x;
            temp.at(2)=y;
            temp.at(3)=z;
            v.push_back(temp);
        }
    }
    if (file_format==2)
    {
        while(!test.eof())
        {
            test >> x;
            test >> y;
            test >> z;
                
            temp.at(0)=0;
            temp.at(1)=x;
            temp.at(2)=y;
            temp.at(3)=z;
                
            v.push_back(temp);
        }
    }
    if (file_format==3)
    {
        double x, y, z, r;
        string s;
        
        test >> s; test >> s; test >> s;
        test >> s; test >> s; test >> s; test >> s;
        test >> s; test >> s; test >> s; test >> s; test >> s;
        test >> s; test >> s; test >> s;
        test >> s; test >> s; test >> s; test >> s;
        test >> s; test >> s; test >> s; test >> s; test >> s;
        test >> s; test >> s; test >> s; test >> s; test >> s;
        test >> s; test >> s; test >> s; test >> s;
        test >> s; test >> s; test >> s; test >> s;
        while(!test.eof())
        {
            test >> x;
            test >> y;
            test >> z;
            test >> r;
            
            temp.at(0)=0;
            temp.at(1)=x;
            temp.at(2)=y;
            temp.at(3)=z;
            
            v.push_back(temp);
        }
    }
        
    row = int(v.size());
    
    if (v.size()>1)
    {
        if (v[int(v.size())-1][0]==v[int(v.size())-2][0] && v[int(v.size())-1][1]==v[int(v.size())-2][1] && v[int(v.size())-1][2]==v[int(v.size())-2][2])
        {
            row = int(v.size());
            v.erase(v.begin()+row-1);
            
        }
    }
        
    cout << v.size() << " monomers...";

    
    return v;
}
//-----------------------------------finds the center of mass-------------------------------
vector<double>center_mass (vector<vector<double> >v)
{
    vector<double>temp(3);
    double mass, xsum=0, ysum=0, zsum=0;
    mass = v.size();
    
    for (int i=0; i<v.size(); i++)
    {
        xsum=xsum+v[i][1];
        ysum=ysum+v[i][2];
        zsum=zsum+v[i][3];
    }
    
    temp.at(0)=xsum/mass;
    temp.at(1)=ysum/mass;
    temp.at(2)=zsum/mass;
    
    return temp;
}
//-----------------------------------recenters cluster-------------------------------
vector<vector<double> >recenter (vector<vector<double> >old, double diam)
{
    double r=diam/2, minimum_x=10E8, minimum_y=10E8, minimum_z=10E8, number1, number2, number3, difference;
    vector<double>COM;
    
    COM = center_mass(old);
    for (int i=0; i<old.size(); i++)
    {
        old[i][1] = old[i][1]-COM[0];
        old[i][2] = old[i][2]-COM[1];
        old[i][3] = old[i][3]-COM[2];
    }
    
    for (int i=0; i<old.size(); i++)
    {
        number1=old[i][1];
        number2=old[i][2];
        number3=old[i][3];
        if (number1<minimum_x)
        {
            minimum_x=number1;
        }
        if (number2<minimum_y)
        {
            minimum_y=number2;
        }
        if (number3<minimum_z)
        {
            minimum_z=number3;
        }
    }
    
    if (minimum_x<r)
    {
        difference = r-minimum_x;
        for (int i=0; i<old.size(); i++)
        {
            old[i][1]=old[i][1]+difference;
        }
    }
    
    if (minimum_y<r)
    {
        difference = r-minimum_y;
        for (int i=0; i<old.size(); i++)
        {
            old[i][2]=old[i][2]+difference;
        }
    }
    if (minimum_z<r)
    {
        difference = r-minimum_z;
        for (int i=0; i<old.size(); i++)
        {
            old[i][3]=old[i][3]+difference;
        }
    }
    return old;
}
//-----------------------------------finds number of dipoles in the space-------------------------------
int max (vector<vector<double> >v)
{
    int number_dipoles, x;
    double maximum=0, number;
    
    x = 2*Rd;
    
    for (int i=0; i<v.size(); i++)
    {
        for (int j=1; j<v[i].size(); j++)
        {
            number = v[i][j];
            if (number>maximum)
            {
                maximum = number;
            }
        }
    }
    number_dipoles = maximum*x+Rd;
    return number_dipoles;
}
//-----------------------------------finds length of each dipole-------------------------------
double length(vector<vector<double> >v, int size)
{
    double length;
    double maximum=0, number;
    
    for (int i=0; i<v.size();i++)
    {
        for (int j=1; j<v[i].size(); j++)
        {
            number = v[i][j];
            number = abs(number);
            if (number>maximum)
            {
                maximum = number;
            }
        }
    }
    
    length = (maximum+radius_monomer)/size;
    
    return length;
}
//----------------------------------Makes room for coating-------------------------------------
vector<vector<double> >buffer(vector<vector<double> >v, double length, int buffer_layers)
{
    for (int i=0; i<v.size(); i++)
    {
        for (int j=1; j<v[i].size(); j++)
        {
            if (mode==0)
            {
                v[i][j]=v[i][j]+(buffer_layers*length);
            }
            else
            {
                v[i][j]=v[i][j]+(buffer_layers*length);
            }
        }
    }
    return v;
}
//-------------------------------------finds which dipoles are core-------------------------------------
void find_core (vector<vector<double> >old, double dcenter, int size, int **shape_file, double **cartesian_space, int **monomer_centers)
{
    double x, y, z, target_x, target_y, target_z, current_x, current_y, current_z, length=dcenter*2;
    int shape_file_rows=0, cartesian_rows=0, ii, jj, kk;
    
    for (int i=0; i<size; i++)
    {
        for (int j=0; j<size; j++)
        {
            for (int k=0; k<size; k++)
            {
                x = i*dcenter*2;
                y = j*dcenter*2;
                z = k*dcenter*2;
                shape_file[shape_file_rows][0]=i;
                shape_file[shape_file_rows][1]=j;
                shape_file[shape_file_rows][2]=k;
                shape_file[shape_file_rows][3]=0;
                cartesian_space[cartesian_rows][0]=x;
                cartesian_space[cartesian_rows][1]=y;
                cartesian_space[cartesian_rows][2]=z;
                shape_file_rows++;
                cartesian_rows++;
            }
        }
    }
    
    cout << endl;
    cout << endl;
    
    int number_monomers=int(old.size());
    for (int i=0; i<number_monomers; i++)
    {
        target_x=old[i][1];
        target_y=old[i][2];
        target_z=old[i][3];
        monomer_centers[i][0] = round(target_x/(dcenter*2));
        monomer_centers[i][1] = round(target_y/(dcenter*2));
        monomer_centers[i][2] = round(target_z/(dcenter*2));
    }
        
    int counter, row1, monomer, row2, row3, row;
    //number_monomers=number_monomers-1;
    monomer = 0;
    do
    {
        if (fmod(double(monomer+1),10)==0)
        {
            cout << monomer+1 << " monomers placed" << endl;
        }
        target_x=old[monomer][1];
        target_y=old[monomer][2];
        target_z=old[monomer][3];
                
        counter=0;
        do
        {
            row1 = counter*size*size;
            current_x=cartesian_space[row1][0];
            if (abs(target_x-current_x)<=length)
            {
                break;
            }
            counter = counter+1;
        }while (counter<size);
        
        counter = 0;
        
        do
        {
            row2=counter*size;
            current_y=cartesian_space[row2][1];
            if(abs(target_y-current_y)<=length)
            {
                break;
            }
            counter=counter+1;
        } while (counter<size);
        
        counter = 0;
        
        do
        {
            row3 = counter;
            current_z=cartesian_space[row3][2];
            if(abs(target_z-current_z)<=length)
            {
                break;
            }
            counter=counter+1;
        } while (counter<size);
        
        
        row=row1+row2+row3;
        double r_particle, r_dipole;
        int firstrow, lastrow;
        
        r_particle = dummy_radius*dummy_radius;
        
        firstrow = row1;
        firstrow = firstrow-(Rd*size*size)-(Rd*size*size)-50;
        if (firstrow<0)
        {
            firstrow=0;
        }
        lastrow = row1;
        lastrow = lastrow+(Rd*size*size)+(Rd*size*size)+50;
        
        firstrow=0;
        lastrow=shape_file_rows-1;
        
        if (lastrow>shape_file_rows-1)
        {
            lastrow=shape_file_rows-1;
        }
        
        for (int i=firstrow; i<=lastrow; i++)
        {
            r_dipole=((cartesian_space[i][0]-target_x)*(cartesian_space[i][0]-target_x))+((cartesian_space[i][1]-target_y)*(cartesian_space[i][1]-target_y))+((cartesian_space[i][2]-target_z)*(cartesian_space[i][2]-target_z));
            
            if (r_dipole<=r_particle)
            {
                if (shape_file[i][3]!=0)
                {
                    //leave empty?
                }
                else
                {
                    shape_file[i][3]=1;
                }
            }
        }
        monomer = monomer+1;
    } while (monomer<number_monomers);
        
    return;
}
//----------------------------------finds mass of aggregate-------------------------------------
double AggregateMass(int **shape_file, int size, double pcore, double pcoating)
{
    double mass;
    double number_of_center=0, number_of_coating=0;
    
    for (int i=0; i<size*size*size; i++)
    {
        if (shape_file[i][3]!=0)
        {
            if (shape_file[i][3]==1)
            {
                number_of_center=number_of_center+1;
            }
            else
            {
                number_of_coating=number_of_coating+1;
            }
        }
    }
    
    mass=(number_of_center*pcore)+(number_of_coating*pcoating);
    
    return mass;
}
//-----------------------------------------Finds aggregate volume-------------------------------------
double AggregateVolume(int **shape_file, double dl, int size)
{
    double volume=0, dv=dl*dl*dl;
    
    for (int i=0; i<size*size*size; i++)
    {
        if (shape_file[i][3]!=0)
        {
            volume=volume+1;
        }
    }
    
    volume=volume*dv;
    
    return volume;
}
//-----------------------------------------Finds core volume-------------------------------------
double CoreVolume(int **shape_file, double dl, int size)
{
    double volume=0, dv=dl*dl*dl;
    
    for (int i=0; i<size*size*size; i++)
    {
        if (shape_file[i][3]==1)
        {
            volume=volume+1;
        }
    }
    
    volume=volume*dv;
    
    return volume;
}
//-------------------------------------finds neighbors-------------------------------------
void neighbors(int **shape_file, int num_dipoles, int *neighbor)
{
    int N;
    
    for (int i=0; i<num_dipoles*num_dipoles*num_dipoles; i++)
    {
        N=0;
        
        if (shape_file[i][3]!=0)
        {
            if (shape_file[i+(num_dipoles*num_dipoles)][3]!=0)
            {
                N++;
            }
            if (shape_file[i-(num_dipoles*num_dipoles)][3]!=0)
            {
                N++;
            }
            if (shape_file[i-num_dipoles][3]!=0)
            {
                N++;
            }
            if (shape_file[i+num_dipoles][3]!=0)
            {
                N++;
            }
            if (shape_file[i-1][3]!=0)
            {
                N++;
            }
            if (shape_file[i+1][3]!=0)
            {
                N++;
            }
        }
        neighbor[i]=N;
    }
    
    return;
}
//----------------------------------coats aggregate-------------------------------------
void coat_even(int **shape_file, int *neighbors, int num_dipoles, double Pcs, double Pcc, int **shape_copy)
{
    double random;
    int seed_placed=-1;
    
    for (int i=0; i<num_dipoles*num_dipoles*num_dipoles; i++)
    {
        shape_copy[i][0]=shape_file[i][0];
        shape_copy[i][1]=shape_file[i][1];
        shape_copy[i][2]=shape_file[i][2];
        shape_copy[i][3]=shape_file[i][3];
    }
    
    for (int i=0; i<num_dipoles*num_dipoles*num_dipoles; i++)
    {
        if (shape_file[i][3]!=0)
        {
            if (neighbors[i]<6)
            {
                if (shape_file[i+(num_dipoles*num_dipoles)][3]==0)
                {
                    random=psdrand(seed);
                    if (shape_copy[i][3]==1 && random<Pcs)
                    {
                        shape_copy[i+(num_dipoles*num_dipoles)][3]=2;
                    }
                    else if (shape_copy[i][3]==2 && random<Pcc)
                    {
                        shape_copy[i+(num_dipoles*num_dipoles)][3]=2;
                    }
                }
                if (shape_file[i-(num_dipoles*num_dipoles)][3]==0)
                {
                    random=psdrand(seed);
                    if (shape_file[i][3]==1 && random<Pcs)
                    {
                        shape_copy[i-(num_dipoles*num_dipoles)][3]=2;
                    }
                    else if (shape_file[i][3]==2 && random<Pcc)
                    {
                        shape_copy[i-(num_dipoles*num_dipoles)][3]=2;
                    }
                }
                if (shape_file[i-num_dipoles][3]==0)
                {
                    random=psdrand(seed);
                    if (shape_file[i][3]==1 && random<Pcs)
                    {
                        shape_copy[i-num_dipoles][3]=2;
                    }
                    else if (shape_file[i][3]==2 && random<Pcc)
                    {
                        shape_copy[i-num_dipoles][3]=2;
                    }
                }
                if (shape_file[i+num_dipoles][3]==0)
                {
                    random=psdrand(seed);
                    if (shape_file[i][3]==1 && random<Pcs)
                    {
                        shape_copy[i+num_dipoles][3]=2;
                    }
                    else if (shape_file[i][3]==2 && random<Pcc)
                    {
                        shape_copy[i+num_dipoles][3]=2;
                    }
                }
                if (shape_file[i-1][3]==0)
                {
                    random=psdrand(seed);
                    if (shape_file[i][3]==1 && random<Pcs)
                    {
                        shape_copy[i-1][3]=2;
                    }
                    else if (shape_file[i][3]==2 && random<Pcc)
                    {
                        shape_copy[i-1][3]=2;
                    }
                }
                if (shape_file[i+1][3]==0)
                {
                    random=psdrand(seed);
                    if (shape_file[i][3]==1 && random<Pcs)
                    {
                        shape_copy[i+1][3]=2;
                    }
                    else if (shape_file[i][3]==2 && random<Pcc)
                    {
                        shape_copy[i+1][3]=2;
                    }
                }
            }
        }
    }
    
    for (int i=0; i<num_dipoles*num_dipoles*num_dipoles; i++)
    {
        shape_file[i][0]=shape_copy[i][0];
        shape_file[i][1]=shape_copy[i][1];
        shape_file[i][2]=shape_copy[i][2];
        shape_file[i][3]=shape_copy[i][3];
    }
    
    return;
}
//----------------------------------coats aggregate junctions-------------------------------------
void coat_junctions(int **shape_file, int** monomer_centers, int num_dipoles, double Pcs, double Pcc, int **shape_copy, int monomers, double gamma)
{
    double dx, dy, dz, dist, pairwise_dist, x, y, z, min, max, x_search, y_search, z_search, r_search;
    int seed_placed=-1, row;
    
    for (int i=0; i<num_dipoles*num_dipoles*num_dipoles; i++)
    {
        shape_copy[i][0]=shape_file[i][0];
        shape_copy[i][1]=shape_file[i][1];
        shape_copy[i][2]=shape_file[i][2];
        shape_copy[i][3]=shape_file[i][3];
    }
        
    for (int i=0; i<monomers; i++)
    {
        for (int j=i+1; j<monomers; j++)
        {
            dx=monomer_centers[i][0]-monomer_centers[j][0];
            dy=monomer_centers[i][1]-monomer_centers[j][1];
            dz=monomer_centers[i][2]-monomer_centers[j][2];
            pairwise_dist = sqrt((dx*dx)+(dy*dy)+(dz*dz));
            if (pairwise_dist<=(8.0/6.0)*2*double(Rd))
            {
                if (abs(dx)>0)
                {
                    if (monomer_centers[i][0] < monomer_centers[j][0])
                    {
                        min = monomer_centers[i][0];
                        max = monomer_centers[j][0];
                    }
                    else
                    {
                        min = monomer_centers[j][0];
                        max = monomer_centers[i][0];
                    }
                    for (int ii=min; ii<=max; ii++)
                    {
                        
                        x = ii;
                        y = monomer_centers[i][1] + (ii-monomer_centers[i][0])*(dy/dx);
                        z = monomer_centers[i][2] + (ii-monomer_centers[i][0])*(dz/dx);
                        for (int jj=int(y-(8.0/6.0)*double(Rd)); jj<int(y+(8.0/6.0)*double(Rd)); jj++)
                        {
                            for (int kk=int(z-(8.0/6.0)*double(Rd)); kk<int(z+(8.0/6.0)*double(Rd)); kk++)
                            {
                                x_search = x;
                                y_search = jj;
                                z_search = kk;
                                r_search = (1-gamma)*(1/(double(Rd)))*((x_search-(min+0.5*(max-min)))*(x_search-(min+0.5*(max-min))))+gamma*(double(Rd));
                                dist = sqrt((x_search-x)*(x_search-x)+(y_search-y)*(y_search-y)+(z_search-z)*(z_search-z));
                                if (dist<=r_search)
                                {
                                    row = x_search*num_dipoles*num_dipoles+y_search*num_dipoles+z_search;
                                    if (shape_file[row][3]==0)
                                    {
                                        shape_copy[row][3]=2;
                                    }
                                }
                            }
                        }
                    }
                }
                
                else if (abs(dy)>0)
                {
                    if (monomer_centers[i][1] < monomer_centers[j][1])
                    {
                        min = monomer_centers[i][1];
                        max = monomer_centers[j][1];
                    }
                    else
                    {
                        min = monomer_centers[j][1];
                        max = monomer_centers[i][1];
                    }
                    for (int ii=min; ii<=max; ii++)
                    {
                        
                        y = ii;
                        x = monomer_centers[i][0] + (ii-monomer_centers[i][1])*(dx/dy);
                        z = monomer_centers[i][2] + (ii-monomer_centers[i][1])*(dz/dy);
                        for (int jj=int(x-(8.0/6.0)*double(Rd)); jj<int(x+(8.0/6.0)*double(Rd)); jj++)
                        {
                            for (int kk=int(z-(8.0/6.0)*double(Rd)); kk<int(z+(8.0/6.0)*double(Rd)); kk++)
                            {
                                x_search = jj;
                                y_search = y;
                                z_search = kk;
                                r_search = (1-gamma)*(1/(double(Rd)))*((y_search-(min+0.5*(max-min)))*(y_search-(min+0.5*(max-min))))+gamma*(double(Rd));
                                dist = sqrt((x_search-x)*(x_search-x)+(y_search-y)*(y_search-y)+(z_search-z)*(z_search-z));
                                if (dist<=r_search)
                                {
                                    row = x_search*num_dipoles*num_dipoles+y_search*num_dipoles+z_search;
                                    if (shape_file[row][3]==0)
                                    {
                                        shape_copy[row][3]=2;
                                    }
                                }
                            }
                        }
                    }
                }
                
                else
                {
                    if (monomer_centers[i][2] < monomer_centers[j][2])
                    {
                        min = monomer_centers[i][2];
                        max = monomer_centers[j][2];
                    }
                    else
                    {
                        min = monomer_centers[j][2];
                        max = monomer_centers[i][2];
                    }
                    for (int ii=min; ii<=max; ii++)
                    {
                        
                        z = ii;
                        x = monomer_centers[i][0] + (ii-monomer_centers[i][2])*(dx/dz);
                        y = monomer_centers[i][1] + (ii-monomer_centers[i][2])*(dy/dz);
                        for (int jj=int(x-(8.0/6.0)*double(Rd)); jj<int(x+(8.0/6.0)*double(Rd)); jj++)
                        {
                            for (int kk=int(y-(8.0/6.0)*double(Rd)); kk<int(y+(8.0/6.0)*double(Rd)); kk++)
                            {
                                x_search = jj;
                                y_search = kk;
                                z_search = z;
                                r_search = (1-gamma)*(1/(double(Rd)))*((z_search-(min+0.5*(max-min)))*(z_search-(min+0.5*(max-min))))+gamma*(double(Rd));
                                dist = sqrt((x_search-x)*(x_search-x)+(y_search-y)*(y_search-y)+(z_search-z)*(z_search-z));
                                if (dist<=r_search)
                                {
                                    row = x_search*num_dipoles*num_dipoles+y_search*num_dipoles+z_search;
                                    if (shape_file[row][3]==0)
                                    {
                                        shape_copy[row][3]=2;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    for (int i=0; i<num_dipoles*num_dipoles*num_dipoles; i++)
    {
        shape_file[i][0]=shape_copy[i][0];
        shape_file[i][1]=shape_copy[i][1];
        shape_file[i][2]=shape_copy[i][2];
        shape_file[i][3]=shape_copy[i][3];
    }
    
    return;
}
//-------------------------------random number generator------------------------------------------
double psdrand(int iseed)
{
    int i, j, k, inx;
    double ran_num;
    static const int ndim = 55, m10 = 1000000000, is = 21, ir = 30;
    static const double base = 1.0E9;
    static int jrand, istack[58];
    static bool init = false;
    
    if((!init) || (iseed < 0))
    {
        iseed = abs(iseed);
        istack[ndim] = iseed;
        j = iseed;
        k = 1;
        
        for(i = 1; i <= (ndim - 1); ++i)
        {
            inx = i*is - int((double)(i*is)/(double)(ndim))*ndim;
            istack[inx] = k;
            k = j - k;
            if(k < 0) {k += m10;}
            j = istack[inx];
        }
        
        for(j = 1; j <= 3; ++j)
        {
            for(i = 1; i <= ndim; ++i)
            {
                inx = i + ir - int((double)(i+ir)/(double)(ndim))*ndim;
                istack[i] -= istack[inx+1];
                if(istack[i] < 0) {istack[i] += m10;}
            }
        }
        jrand = 0;
        init = true;
    }
    
    jrand += 1;
    
    if(jrand > ndim)
    {
        for(i = 1; i <= ndim; ++i)
        {
            inx = i + ir - ((int)((double)(i+ir)/(double)(ndim)))*ndim;
            istack[i] -= istack[inx+1];
            if(istack[i] < 0) {istack[i] += m10;}
        }
        jrand = 1;
    }
    
    ran_num = ((double)istack[jrand]) / base;
    
    return ran_num;
}
//----------------------------------finds cross-sectional area-------------------------------------
vector<double>area(int **shape_file, int max, int **shape_copy)
{
    vector<double>v(6);
    int i=0, i2;
    double sum=0, counter1=0, counter2, counter3=0;
    
    for (int i=0; i<(max*max*max)-1; i++) //reset copy
    {
        shape_copy[i][0]=shape_file[i][0];
        shape_copy[i][1]=shape_file[i][1];
        shape_copy[i][2]=shape_file[i][2];
        shape_copy[i][3]=shape_file[i][3];
    }
    
    do      //xy plane sweeping up, just core
    {
        counter1=0;
        sum=0;
        
        do
        {
            i=counter3+counter1*max*max;
            counter2=1;
            do
            {
                if (shape_file[i][3]==1)
                {
                    i2=i;
                    do
                    {
                        shape_copy[i2][3]=1;
                        i2=i2+1;
                    } while (shape_copy[i2][2]!=max-1);
                    shape_copy[i2][3]=1;
                }
                counter2=counter2+1;
                i=i+max;
            } while (counter2<=max);
            counter1=counter1+1;
        }while (counter1<max);
        counter3=counter3+1;
    }while (counter3<max);
    
    sum = 0;
    
    for (int i=0; i<max*max*max; i++)
    {
        if (shape_copy[i][2]==max-1 && shape_copy[i][3]==1)
        {
            sum = sum+1;
        }
    }
    
    v.at(0)=sum;
    
    
    for (int i=0; i<(max*max*max)-1; i++) //reset copy
    {
        shape_copy[i][0]=shape_file[i][0];
        shape_copy[i][1]=shape_file[i][1];
        shape_copy[i][2]=shape_file[i][2];
        shape_copy[i][3]=shape_file[i][3];
    }
    
    counter3=0;
    
    do      //xz plane sweeping out, just core
    {
        counter1=0;
        sum=0;
        
        do
        {
            i=counter3+counter1*max*max;
            counter2=1;
            do
            {
                if (shape_file[i][3]==1)
                {
                    i2=i;
                    do
                    {
                        shape_copy[i2][3]=1;
                        i2=i2+max;
                    } while (shape_copy[i2][1]!=max-1);
                    shape_copy[i2][3]=1;
                }
                counter2=counter2+1;
                i=i+max;
            } while (counter2<=max);
            counter1=counter1+1;
        }while (counter1<max);
        counter3=counter3+1;
    }while (counter3<max);
    
    for (int i=0; i<max*max*max; i++)
    {
        if (shape_copy[i][1]==max-1 && shape_copy[i][3]==1)
        {
            sum = sum+1;
        }
    }
    
    v.at(1)=sum;
    
    for (int i=0; i<(max*max*max)-1; i++) //reset copy
    {
        shape_copy[i][0]=shape_file[i][0];
        shape_copy[i][1]=shape_file[i][1];
        shape_copy[i][2]=shape_file[i][2];
        shape_copy[i][3]=shape_file[i][3];
    }
    
    counter3=0;
    
    do      //yz plane sweeping out, just core
    {
        counter1=0;
        sum=0;
        
        do
        {
            i=counter3+counter1*max*max;
            counter2=1;
            do
            {
                if (shape_file[i][3]==1)
                {
                    i2=i;
                    do
                    {
                        shape_copy[i2][3]=1;
                        i2=i2+max*max;
                    } while (shape_copy[i2][0]!=max-1);
                    shape_copy[i2][3]=1;
                }
                counter2=counter2+1;
                i=i+max;
            } while (counter2<=max);
            counter1=counter1+1;
        }while (counter1<max);
        counter3=counter3+1;
    }while (counter3<max);
    
    for (int i=0; i<max*max*max; i++)
    {
        if (shape_copy[i][0]==max-1 && shape_copy[i][3]==1)
        {
            sum = sum+1;
        }
    }
    
    v.at(2)=sum;
    
    for (int i=0; i<(max*max*max)-1; i++) //reset copy
    {
        shape_copy[i][0]=shape_file[i][0];
        shape_copy[i][1]=shape_file[i][1];
        shape_copy[i][2]=shape_file[i][2];
        shape_copy[i][3]=shape_file[i][3];
    }
    
    counter3=0;
    
    do      //xy plane sweeping up
    {
        counter1=0;
        sum=0;
        
        do
        {
            i=counter3+counter1*max*max;
            counter2=1;
            do
            {
                if (shape_file[i][3]!=0)
                {
                    i2=i;
                    do
                    {
                        shape_copy[i2][3]=1;
                        i2=i2+1;
                    } while (shape_copy[i2][2]!=max-1);
                    shape_copy[i2][3]=1;
                }
                counter2=counter2+1;
                i=i+max;
            } while (counter2<=max);
            counter1=counter1+1;
        }while (counter1<max);
        counter3=counter3+1;
    }while (counter3<max);
    
    sum = 0;
    
    for (int i=0; i<max*max*max; i++)
    {
        if (shape_copy[i][2]==max-1 && shape_copy[i][3]!=0)
        {
            sum = sum+1;
        }
    }
    
    v.at(3)=sum;

    
    for (int i=0; i<(max*max*max)-1; i++) //reset copy
    {
        shape_copy[i][0]=shape_file[i][0];
        shape_copy[i][1]=shape_file[i][1];
        shape_copy[i][2]=shape_file[i][2];
        shape_copy[i][3]=shape_file[i][3];
    }
    
    counter3=0;
    
    do      //xz plane sweeping out
    {
        counter1=0;
        sum=0;
        
        do
        {
            i=counter3+counter1*max*max;
            counter2=1;
            do
            {
                if (shape_file[i][3]!=0)
                {
                    i2=i;
                    do
                    {
                        shape_copy[i2][3]=1;
                        i2=i2+max;
                    } while (shape_copy[i2][1]!=max-1);
                    shape_copy[i2][3]=1;
                }
                counter2=counter2+1;
                i=i+max;
            } while (counter2<=max);
            counter1=counter1+1;
        }while (counter1<max);
        counter3=counter3+1;
    }while (counter3<max);
    
    for (int i=0; i<max*max*max; i++)
    {
        if (shape_copy[i][1]==max-1 && shape_copy[i][3]!=0)
        {
            sum = sum+1;
        }
    }
    
    v.at(4)=sum;

    for (int i=0; i<(max*max*max)-1; i++) //reset copy
    {
        shape_copy[i][0]=shape_file[i][0];
        shape_copy[i][1]=shape_file[i][1];
        shape_copy[i][2]=shape_file[i][2];
        shape_copy[i][3]=shape_file[i][3];
    }
    
    counter3=0;
    
    do      //yz plane sweeping out
    {
        counter1=0;
        sum=0;
        
        do
        {
            i=counter3+counter1*max*max;
            counter2=1;
            do
            {
                if (shape_file[i][3]!=0)
                {
                    i2=i;
                    do
                    {
                        shape_copy[i2][3]=1;
                        i2=i2+max*max;
                    } while (shape_copy[i2][0]!=max-1);
                    shape_copy[i2][3]=1;
                }
                counter2=counter2+1;
                i=i+max;
            } while (counter2<=max);
            counter1=counter1+1;
        }while (counter1<max);
        counter3=counter3+1;
    }while (counter3<max);
    
    for (int i=0; i<max*max*max; i++)
    {
        if (shape_copy[i][0]==max-1 && shape_copy[i][3]!=0)
        {
            sum = sum+1;
        }
    }
    
    v.at(5)=sum;
    
    return v;
}
//----------------------------------finds SA fraction-------------------------------------
double SA_fraction(int **shape_file, int num_dipoles)
{
    double output=0, SA_bare=0, SA_coated=0;
    
    for (int i=0; i<num_dipoles*num_dipoles*num_dipoles; i++)
    {
        if (shape_file[i][3]==1)
        {
            if (shape_file[i+(num_dipoles*num_dipoles)][3]==0)
            {
                SA_bare=SA_bare+1;
            }
            else if (shape_file[i+(num_dipoles*num_dipoles)][3]==2)
            {
                SA_bare=SA_bare+1;
                SA_coated=SA_coated+1;
            }
            if (shape_file[i-(num_dipoles*num_dipoles)][3]==0)
            {
                SA_bare=SA_bare+1;
            }
            else if (shape_file[i-(num_dipoles*num_dipoles)][3]==2)
            {
                SA_bare=SA_bare+1;
                SA_coated=SA_coated+1;
            }
            if (shape_file[i-num_dipoles][3]==0)
            {
                SA_bare=SA_bare+1;
            }
            else if (shape_file[i-num_dipoles][3]==2)
            {
                SA_bare=SA_bare+1;
                SA_coated=SA_coated+1;
            }
            if (shape_file[i+num_dipoles][3]==0)
            {
                SA_bare=SA_bare+1;
            }
            else if (shape_file[i+num_dipoles][3]==2)
            {
                SA_bare=SA_bare+1;
                SA_coated=SA_coated+1;
            }
            if (shape_file[i-1][3]==0)
            {
                SA_bare=SA_bare+1;
            }
            else if (shape_file[i-1][3]==2)
            {
                SA_bare=SA_bare+1;
                SA_coated=SA_coated+1;
            }
            if (shape_file[i+1][3]==0)
            {
                SA_bare=SA_bare+1;
            }
            else if (shape_file[i+1][3]==2)
            {
                SA_bare=SA_bare+1;
                SA_coated=SA_coated+1;
            }
            //cout << SA_coated << " " << SA_bare << endl;
        }
    }
    
    output=SA_coated/SA_bare;
    
    return output;
}

