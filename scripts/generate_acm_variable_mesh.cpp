/**
 * @file generate_acm_variable_mesh.cpp
 * @brief Generate small Cartesian 2-D/3-D CGNS meshes for independent ACMVariable verification.
 * @author Runzhi Ma
 * @date 2026-09-03
 * @details Usage: executable output.cgns dim subdivisions. Domain is [0,1]^dim;
 * all external faces are the named FAR zone. Existing project meshes are never edited.
 */
#include <cgnslib.h>
#include <vector>
#include <string>
#include <stdexcept>
#include <iostream>

/** @brief Enforce CGNS success. @param code CGNS return status. */
static void Check(int code) { if (code) throw std::runtime_error(cg_get_error()); }

/** @brief Generate mesh. @param argc Argument count. @param argv Path/dim/resolution.
 * @return Zero on success, one on invalid arguments or CGNS failure. */
int main(int argc, char **argv)
{
    try
    {
        if (argc != 4) throw std::runtime_error("usage: output.cgns dim n");
        const int dim = std::stoi(argv[2]), n = std::stoi(argv[3]);
        if ((dim != 2 && dim != 3) || n < 2) throw std::runtime_error("dim=2/3 and n>=2 required");
        const int nz = dim == 2 ? 0 : n;
        auto node = [=](int i, int j, int k) -> cgsize_t
        { return 1 + i + (n+1) * (j + (n+1)*k); };
        std::vector<double> x,y,z;
        for (int k=0;k<=nz;++k) for (int j=0;j<=n;++j) for (int i=0;i<=n;++i)
        { x.push_back(double(i)/n); y.push_back(double(j)/n); z.push_back(double(k)/n); }
        std::vector<cgsize_t> volume, wall;
        if (dim == 2)
        {
            for (int j=0;j<n;++j) for (int i=0;i<n;++i)
                for (auto v : {node(i,j,0),node(i+1,j,0),node(i+1,j+1,0),node(i,j+1,0)})
                    volume.push_back(v);
            for (int i=0;i<n;++i)
                for (auto v : {node(i,0,0),node(i+1,0,0),node(i+1,n,0),node(i,n,0),
                               node(0,i+1,0),node(0,i,0),node(n,i,0),node(n,i+1,0)})
                    wall.push_back(v);
        }
        else
        {
            for (int k=0;k<n;++k) for (int j=0;j<n;++j) for (int i=0;i<n;++i)
                for (auto v : {node(i,j,k),node(i+1,j,k),node(i+1,j+1,k),node(i,j+1,k),
                               node(i,j,k+1),node(i+1,j,k+1),node(i+1,j+1,k+1),node(i,j+1,k+1)})
                    volume.push_back(v);
            for (int j=0;j<n;++j) for (int i=0;i<n;++i)
            {
                for (auto v : {node(i,j,0),node(i,j+1,0),node(i+1,j+1,0),node(i+1,j,0),
                               node(i,j,n),node(i+1,j,n),node(i+1,j+1,n),node(i,j+1,n),
                               node(0,i,j),node(0,i,j+1),node(0,i+1,j+1),node(0,i+1,j),
                               node(n,i,j),node(n,i+1,j),node(n,i+1,j+1),node(n,i,j+1),
                               node(i,0,j),node(i+1,0,j),node(i+1,0,j+1),node(i,0,j+1),
                               node(i,n,j),node(i,n,j+1),node(i+1,n,j+1),node(i+1,n,j)})
                    wall.push_back(v);
            }
        }
        int file,base,zone,section,coord,boco;
        Check(cg_open(argv[1],CG_MODE_WRITE,&file));
        Check(cg_base_write(file,"Base",dim,dim,&base));
        const cgsize_t cells = volume.size() / (dim==2 ? 4 : 8);
        const cgsize_t faces = wall.size() / (dim==2 ? 2 : 4);
        cgsize_t size[3] = {static_cast<cgsize_t>(x.size()),cells,0};
        Check(cg_zone_write(file,base,"Cartesian",size,Unstructured,&zone));
        Check(cg_coord_write(file,base,zone,RealDouble,"CoordinateX",x.data(),&coord));
        Check(cg_coord_write(file,base,zone,RealDouble,"CoordinateY",y.data(),&coord));
        if(dim==3) Check(cg_coord_write(file,base,zone,RealDouble,"CoordinateZ",z.data(),&coord));
        Check(cg_section_write(file,base,zone,"Cells",dim==2?QUAD_4:HEXA_8,1,cells,0,volume.data(),&section));
        Check(cg_section_write(file,base,zone,"BoundaryFaces",dim==2?BAR_2:QUAD_4,
                               cells+1,cells+faces,0,wall.data(),&section));
        cgsize_t range[2]={cells+1,cells+faces};
        Check(cg_boco_write(file,base,zone,"FAR",BCFarfield,PointRange,2,range,&boco));
        Check(cg_boco_gridlocation_write(file,base,zone,boco,dim==2?EdgeCenter:FaceCenter));
        Check(cg_close(file));
        std::cout << "Created " << cells << " cells and " << faces << " boundary faces\n";
    }
    catch(const std::exception &e) { std::cerr << e.what() << '\n'; return 1; }
}
