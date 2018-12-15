#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <ios>
#include <vector>
#include <set>
#include <cuda_runtime.h>
#include <vector_types.h>
#include <device_launch_parameters.h>
#include <curand.h>
#include <curand_kernel.h>
#include <chrono>
#include <float.h>
#include "lodepng.h" // super easy read and write for png images
#include "cutil_math.h"

#define DISPERSION

// return min and max components of a vector
inline __host__ __device__ float3 minf3(float3 a, float3 b){
    return make_float3(a.x<b.x?a.x:b.x, a.y<b.y?a.y:b.y, a.z<b.z?a.z:b.z);
}
inline __host__ __device__ float3 maxf3(float3 a, float3 b){
    return make_float3(a.x>b.x?a.x:b.x, a.y>b.y?a.y:b.y, a.z>b.z?a.z:b.z);
}

//min and max of two floats
inline __device__ float minf(float a, float b){return a<b?a:b;}
inline __device__ float maxf(float a, float b){return a>b?a:b;}

// in case something goes wrong
void print_error(std::string message){
    std::cerr << "Error: " << message << std::endl;
    exit(1);
}

// variables on host
int h_width;
int h_height;
int h_samples;
int bloom_rad;

__constant__ float3* textures;

#include "geometry.h"
#include "bvh.h"
#include "read_scene.h"

// variables and geometry on device
__constant__ int width;
__constant__ int height;
__constant__ int samples;
__constant__ const_text* const_textures;
__constant__ check_text* check_textures;
__constant__ pix_text* pix_textures;
__constant__ Sphere* spheres;
__constant__ Box* boxes;
__constant__ Triangle* all_triangles;
__constant__ uint num_triangles;
__constant__ compact_BVHnode* bvh;
__constant__ uint num_nodes;
__constant__ uint num_spheres;
__constant__ uint num_boxes;

// copy scene info to device
void init_scene(){
    read_scene(); // read from file
    
    create_compact_BVH();

    const_text* d_const;
    cudaMalloc((void**)(&d_const), h_const.size()*sizeof(const_text));
    cudaMemcpy((void*)(d_const), (void*)(&(h_const[0])),
	       h_const.size()*sizeof(const_text), cudaMemcpyHostToDevice);
    check_text* d_check;
    cudaMalloc((void**)(&d_check), h_check.size()*sizeof(check_text));
    cudaMemcpy((void*)(d_check), (void*)(&(h_check[0])),
	       h_check.size()*sizeof(check_text), cudaMemcpyHostToDevice);
    pix_text* d_pix;
    cudaMalloc((void**)(&d_pix), h_pix.size()*sizeof(pix_text));
    cudaMemcpy((void*)(d_pix), (void*)(&(h_pix[0])), h_pix.size()*sizeof(pix_text),
	       cudaMemcpyHostToDevice);
    float3* d_text;
    cudaMalloc(&d_text, h_textures.size()*sizeof(float3));
    cudaMemcpy(d_text, &(h_textures[0]), h_textures.size()*sizeof(float3),
	       cudaMemcpyHostToDevice);
    Sphere* d_spheres;
    cudaMalloc((void**)(&d_spheres), h_spheres.size()*sizeof(Sphere));
    cudaMemcpy((void*)(d_spheres), (void*)(&(h_spheres[0])),
	       h_spheres.size()*sizeof(Sphere), cudaMemcpyHostToDevice);
    Box* d_boxes;
    cudaMalloc((void**)(&d_boxes), h_boxes.size()*sizeof(Box));
    cudaMemcpy((void*)(d_boxes), (void*)(&(h_boxes[0])), h_boxes.size()*sizeof(Box),
	       cudaMemcpyHostToDevice);
    Triangle* d_triangles;
    cudaMalloc((void**)(&d_triangles), triangles_ordered.size()*sizeof(Triangle));
    cudaMemcpy((void*)(d_triangles), (void*)(&(triangles_ordered[0])),
	       triangles_ordered.size()*sizeof(Triangle), cudaMemcpyHostToDevice);
    compact_BVHnode* d_bvh;
    cudaMalloc((void**)(&d_bvh), compact_BVH.size()*sizeof(compact_BVHnode));
    cudaMemcpy((void*)(d_bvh), (void*)(&(compact_BVH[0])),
	       compact_BVH.size()*sizeof(compact_BVHnode), cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(const_textures, &d_const, sizeof(const_text*));
    cudaMemcpyToSymbol(check_textures, &d_check, sizeof(check_text*));
    cudaMemcpyToSymbol(pix_textures, &d_pix, sizeof(pix_text*));
    cudaMemcpyToSymbol(textures, &d_text, sizeof(float3*));
    cudaMemcpyToSymbol(spheres, &d_spheres, sizeof(Sphere*));
    cudaMemcpyToSymbol(boxes, &d_boxes, sizeof(Box*));
    cudaMemcpyToSymbol(all_triangles, &d_triangles, sizeof(Triangle*));
    cudaMemcpyToSymbol(bvh, &d_bvh, sizeof(compact_BVHnode*));

    uint* d_spheres_size;
    uint* d_boxes_size;
    uint* d_triangles_size;
    uint* d_bvh_size;
    uint ss = h_spheres.size();
    uint sb = h_boxes.size();
    uint st = triangles_ordered.size();
    uint sh = compact_BVH.size();
    cudaMalloc((void**)(&d_spheres_size), sizeof(int));
    cudaMalloc((void**)(&d_boxes_size), sizeof(int));
    cudaMalloc((void**)(&d_triangles_size), sizeof(int));
    cudaMalloc((void**)(&d_bvh_size), sizeof(int));
    cudaMemcpy((void*)d_spheres_size, &ss, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy((void*)d_boxes_size, &sb, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy((void*)d_triangles_size, &st, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy((void*)d_bvh_size, &sh, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpyToSymbol(num_spheres, d_spheres_size, sizeof(int));
    cudaMemcpyToSymbol(num_boxes, d_boxes_size, sizeof(int));
    cudaMemcpyToSymbol(num_triangles, d_triangles_size, sizeof(int));
    cudaMemcpyToSymbol(num_nodes, d_bvh_size, sizeof(int));

    uint* d_width;
    uint* d_height;
    uint* d_samples;
    cudaMalloc((void**)(&d_width), sizeof(int));
    cudaMalloc((void**)(&d_height), sizeof(int));
    cudaMalloc((void**)(&d_samples), sizeof(int));
    cudaMemcpy(d_width, &h_width, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_height, &h_height, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_samples, &h_samples, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(width, d_width, sizeof(int));
    cudaMemcpyToSymbol(height, d_height, sizeof(int));
    cudaMemcpyToSymbol(samples, d_samples, sizeof(int));
    std::cout << "Initialized " << triangles_ordered.size() << " triangles" << std::endl; 
}

// recursively intersect ray with bvh to find triangle in sublinear time
// I really need to give each mesh a separate bvh but I haven't yet
__device__ void intersect_triangles(const Ray& r_in, float& t, float& dist, int& id){
    int stack[64]; // its reasonable to assume this will be way bigger than neccesary
    int stack_idx = 1;
    stack[0] = 0;
    float d;
    float tb = -1e20; // large negative
    while(stack_idx){
	int boxidx = stack[stack_idx - 1]; // pop off top of stack
	stack_idx --; 
	if(!(bvh[boxidx].u.leaf.count & 0x80000000)){ // inner
	    Box b;
	    b.min = bvh[boxidx].min;
	    b.max = bvh[boxidx].max;
	    if (b.intersect(r_in)){
		stack[stack_idx++] = bvh[boxidx].u.inner.left; // push right and left onto stack
		stack[stack_idx++] = bvh[boxidx].u.inner.right;
	    }
	} 
	else{ // leaf
	    for (int i = bvh[boxidx].u.leaf.offset;
		 i < bvh[boxidx].u.leaf.offset + (bvh[boxidx].u.leaf.count & 0x7fffffff);
		 i++){ // intersect all triangles in this box
		if ((d = all_triangles[i].intersect(r_in)) && d > -1e19){
		    if(d<t && d>0.001){
			t=d;
			id = i;
		    }
		    else if(d>tb && d<0.001){
			tb = d;
		    }
		}
	    }
	}
    }
    dist = t - tb;
    if (tb < -1e19) dist = 0; // nothing intersected
}

// find first thing that the ray hits
__device__ bool intersect_scene(const Ray& r_in, Hit_data& dat){
    int n = num_spheres;
    float t = 1e20;
    float d;
    int id;

    dat.dist = 1;

    dat.t = 1e20;
    float dist, dist_save;
    for(int i = int(n); i--; ){ // intersect spheres
	if ((d = spheres[i].intersect(r_in, dist)) && d<t && d > 0){
	    t = d;
	    id = i;
	    dist_save = dist;
	}
    }
    // only update dat at most once per geometry type
    if (t < dat.t){
	dat.p = r_in.origin + t*r_in.direction;
	dat.t = t;
	dat.scat = spheres[id].scat;
	dat.text = spheres[id].text;
	dat.normal = normalize(dat.p - spheres[id].center);
	dat.texture_num = spheres[id].texture_num;
	dat.uv = make_float2(0.5 + atan(dat.normal.z/dat.normal.x)/2/M_PI,
			     0.5 - asin(dat.normal.y)/2/M_PI);
	dat.shift = spheres[id].radius/100000;
	t = dat.t;
	if (dist_save < 1e19)
	    dat.dist = dist_save;
	else
	    dat.dist = 0; // shouldn't happen. means nothing hit
    }
    
    n = num_boxes;
    for(int i = int(n); i--; ){ // intersect boxes
	if ((d = boxes[i].intersect(r_in)) && d<t && d>0){
	    t=d;
	    id = i;
	}
    }

    if (t<dat.t){
	dat.p = r_in.origin + t*r_in.direction;
	dat.t = t;
	dat.scat = boxes[id].scat;
	dat.text = boxes[id].text;
	dat.normal = boxes[id].normal_at(r_in.origin + t*r_in.direction);
	dat.texture_num = boxes[id].texture_num;
	dat.uv = boxes[id].uv_at(dat.p);
	dat.shift = 0;
	t = dat.t;
    }
    
    intersect_triangles(r_in, t, dist, id);
    
    if (t<dat.t){
	dat.p = r_in.origin + t*r_in.direction;
	dat.t = t;
	dat.scat = all_triangles[id].scat;
	dat.text = all_triangles[id].text;
	dat.normal = all_triangles[id].normal;
	dat.texture_num = all_triangles[id].texture_num;
	dat.uv = all_triangles[id].uv_at(dat.p);
	dat.dist = dist;
	dat.shift = 0;
    }  
    
    return dat.t<1e20; // return whether anying was hit
}

__device__ float3 radiance(Ray& r_in, curandState* randstate){
    float3 color = make_float3(0.0, 0.0, 0.0); //accumulated total light
    float3 mask; //accumulated color mask
    float idx_refr;
#ifdef DISPERSION
    // do each rgb separately
    // very subtle for most scenes and very slow
    if (r_in.light == RED){
	mask = make_float3(1.0,0.0,0.0);
	idx_refr = 1.5;
    }
    else if (r_in.light == GREEN){
	mask = make_float3(0.0,1.0,0.0);
	idx_refr = 1.52;
    }
    else{
	mask = make_float3(0.0,0.0,1.0);
	idx_refr = 1.54;
    }
#else
    mask = make_float3(1.0,1.0,1.0);
    idx_refr = 1.5;
#endif

    // loop is faster than recursion and easy to do here
    for (int bounces = 0; bounces < 40; ++bounces){
	Hit_data dat;
	if(!intersect_scene(r_in, dat)){ // if ray doesn't hit anything
	    float t = (r_in.direction.y + 1)/2;
	    return color + mask*(make_float3(1)*(1-t) + make_float3(0.5,0.7,1)*t); // gradient sky
	}
	// normal pointing in correct way
	float3 nl = dot(dat.normal, r_in.direction) < 0 ? dat.normal : dat.normal*-1;
	float3 d;

	float3 col; 
	float3 emission;
	float dist_traveled = dat.t; // for everything except volumes

	// color doesn't depend on point
	if(dat.text == CONST){
	    col = const_textures[dat.texture_num].color;
	    emission = const_textures[dat.texture_num].emission;
	}

	// checkered using uv map
	if(dat.text == CHECK){
	    check_text ct = check_textures[dat.texture_num];
	    emission = ct.emission;
	    float n = sinf(dat.uv.x*ct.scale.x)*sinf(dat.uv.y*ct.scale.y);
	    if (n<0) col = ct.color1;
	    else col = ct.color2;
	}
	
	// image texture using uv
	if(dat.text == PIX){
	    pix_text pt = pix_textures[dat.texture_num];
	    col = pt.color_at(dat.uv);
	}	

	// perfectly diffuse
	if(dat.scat == DIFF){\
	    // pick random direction
	    float theta = 2*M_PI*curand_uniform(randstate);
	    float cosphi = curand_uniform(randstate);
	    float sinphi = sqrtf(1-cosphi*cosphi);
	    
	    float3 w = nl;
	    float3 u = normalize(cross((fabs(w.x) > 0.0001 ?
					make_float3(0,1,0) :
					make_float3(1,0,0)), w));
	    float3 v = cross(w,u);
	    
	    //rotate new ray to correct hemisphere
	    d  = normalize(u*cosf(theta)*sinphi +
			   v*sinf(theta)*sinphi +
			   w*cosphi);
	}	

	// perfectly specular reflection
	if (dat.scat == SPEC){
	    d = r_in.direction - 2.0*dat.normal*dot(dat.normal,r_in.direction);
	}

	// either perfect reflection or perfect refraction depending on Schlick's approxomation
	if (dat.scat == REFR){
	    float3 reflected = r_in.direction - 2.0*dat.normal*dot(dat.normal,r_in.direction);
	    float ratio;
	    float3 refracted;
	    float reflect_prob;
	    float cosine;
	    if(dot(r_in.direction, dat.normal) > 0){// ray is entering
		ratio = idx_refr;
		cosine = idx_refr*dot(r_in.direction, dat.normal);
	    }
	    else{ // ray is leaving
		ratio = 1.0/idx_refr;
		cosine = -dot(r_in.direction, dat.normal);
	    }

	    // Schlick's approxomation
	    float dt = dot(r_in.direction, nl);
	    float disc = 1.0 - ratio*ratio*(1-dt*dt);
	    if(disc > 0){
		refracted = ratio*(r_in.direction - dt*nl) - sqrtf(disc)*nl;
		float r0 = (1 - idx_refr) / (1 + idx_refr);
		r0 = r0*r0;
		float c = 1-cosine;
		reflect_prob = r0+(1-r0)*c*c*c*c*c;
	    }
	    else{
		reflect_prob = 1.0;
	    }
	    
	    if(curand_uniform(randstate) < reflect_prob)
		d = reflected;
	    else{
		d = refracted;
		// change media attenuation color
		if (dat.dist <  0.01){ //entering
		    r_in.attenuation.x = col.x;
		    r_in.attenuation.y = col.y;
		    r_in.attenuation.z = col.z;
		}
		else{//leaving
		    r_in.attenuation.x = 1;
		    r_in.attenuation.y = 1;
		    r_in.attenuation.z = 1;
		}
		dat.shift = 0;	
	    }
	    col = make_float3(1);
	}

	// metal material from Peter Shirley's Ray Tracing in one Weekend
	if(dat.scat == METAL){
	    float phi = 2*M_PI*curand_uniform(randstate);
	    float r = curand_uniform(randstate);
	    float exponent = 10;
	    float cosTheta = powf(1-r, 1.0/(exponent+1));
	    float sinTheta = sqrtf(1-cosTheta*cosTheta);

	    float3 w = normalize(r_in.direction - 2.0*dat.normal*dot(dat.normal,r_in.direction));
	    float3 u = cross((fabs(w.x) > 0.001?make_float3(0,1,0):make_float3(1,0,0)),w);
	    float3 v = cross(w,u);

	    d = normalize(u*cosf(phi)*sinTheta + v*sinf(phi)*sinTheta + w*cosTheta);
	}

	// like glass except diffuse instead of refract
	// looks like glazed ceramic
	if (dat.scat == COAT){
	    // Schlick's approxomation
	    float c = 1 + idx_refr*dot(r_in.direction, nl);
	    float r0 = (1 - idx_refr) / (1 + idx_refr);
	    r0 = r0*r0;
	    float reflect_prob = r0+(1-r0)*c*c*c*c*c;
	    
	    if(curand_uniform(randstate)<reflect_prob){
		d = r_in.direction - 2.0*dat.normal*dot(dat.normal,r_in.direction);
		col = make_float3(1);
	    }
	    else{
		float theta = 2*M_PI*curand_uniform(randstate);
		float cosphi = curand_uniform(randstate);
		float sinphi = sqrtf(1-cosphi*cosphi);
	    
		float3 w = nl;
		float3 u = normalize(cross((fabs(w.x) > 0.0001 ?
					    make_float3(0,1,0) :
					    make_float3(1,0,0)), w));
		float3 v = cross(w,u);

		d  = normalize(u*cosf(theta)*sinphi +
			       v*sinf(theta)*sinphi +
			       w*cosphi);	
	    }
	}

	// calculate distance traveled based on particle density
	if (dat.scat == VOLUME){
	    float density = const_textures[dat.texture_num].density;
	    float dist = -(1/density)*logf(curand_uniform(randstate));
	    if(dat.dist > 0){ // ray started inside
		if (dist < dat.t){ // ray ends inside
		    // reflect in random direction
		    float theta = curand_uniform(randstate)*2*M_PI;
		    float cosphi = 2*curand_uniform(randstate) - 1;
		    float sinphi = sqrtf(1 - cosphi*cosphi);
		    d = normalize(make_float3(cosf(theta)*sinphi,
					      sinf(theta)*sinphi,
					      cosphi));
		    dat.p = r_in.origin + dist*r_in.direction; // origin of new ray
		    dist_traveled = dist; // ray didn't go all the way to intersection
		}
		else{
		    // continue in same direction
		    d = r_in.direction;
		    col = make_float3(1);
		}
	    }
	    else{ // ray started outside
		d = r_in.direction;
		col = make_float3(1);
	    }
	    dat.shift = 0; // dont move ray origin
	}

	// attenuation due to media
	float r = __expf(-1*(1-r_in.attenuation.x)*dist_traveled);
	float g = __expf(-1*(1-r_in.attenuation.y)*dist_traveled);
	float b = __expf(-1*(1-r_in.attenuation.z)*dist_traveled);
	mask = mask*make_float3(r,g,b);

	color += mask*emission; // if object emits light add to total
	
	mask = mask * col; // update color mask

	// new ray
	r_in.origin = dat.p + nl*dat.shift;
	r_in.direction = d;

	if (mask.x + mask.y + mask.z < 0.01) break; // if ray has lost most energy no need to continue
	
    }
    
    return color;
}

// call from host
__global__ void render_kernel(float3 *output){
    // coordinates of pixel
    uint x = blockIdx.x*blockDim.x + threadIdx.x;
    uint y = blockIdx.y*blockDim.y + threadIdx.y;

    // index in output array
    uint i = (height - y - 1)*width + x;

    uint threadId = (blockIdx.x+blockIdx.y*gridDim.x)*
	(blockDim.x*blockDim.y) + (threadIdx.y*blockDim.x) + threadIdx.x;
    
    curandState randstate;
    curand_init(threadId,0,0,&randstate); //initialize host

    //eventually I would like to make this interactive so this camera stuff is here
    float3 cam = make_float3(0,15,40);
    float3 look_at = make_float3(0,15,0);
    float3 w = normalize(cam - look_at);
    float3 u = cross(make_float3(0,1,0),w);
    float3 v = cross(w,u);
    float focal_length = length(cam - look_at);
    float aperture = 20.25;
    float lens_radius = 0.5;
    float he = tanf(aperture/2);
    float wi = he*width/height;
    
    float3 screen_corner = cam - wi*focal_length*u - he*focal_length*v - focal_length*w;
    float3 horiz = 2*wi*focal_length*u;
    float3 vert = 2*he*focal_length*v;
    float3 r = make_float3(0.0);

    float3 d;
    for (int s = 0; s<samples; ++s){ // new ray for each sample
	float theta = 2*M_PI*curand_uniform(&randstate);
	float rad = lens_radius*curand_uniform(&randstate);
	float3 from = cam + rad*(u*cosf(theta) + v*sinf(theta)); // slight random for depth of field

	float xs = curand_uniform(&randstate);
	float ys = curand_uniform(&randstate);
	d = normalize(screen_corner + horiz*(float(x+xs)/float(width)) +
		      vert*(float(y+ys)/float(height)) - from);
	light_type l = ((s%3==0)?RED:((s%3==1)?GREEN:BLUE)); // only used for dispersion
	Ray init_ray = Ray(from, d, make_float3(1,1,1), l);
	r = r + radiance(init_ray,&randstate);
    }
    r = r/samples;
#ifdef DISPERSION
    r = r*3;
#endif
    output[i] = r;
}

// utilities to turn light vals into image
inline float clamp(float x){return x<0.0? 0.0: x>1.0 ? 1.0 : x;}
inline int to_int(float x){return int(pow(clamp(x), 1/2.2)*255 + 0.5);}

// apply gausian blur to very bright pixels to get a bloom effect
void bloom(float3* in, float3* out, int radius, float stddev){
    float kernel[(2*radius+1)*(2*radius+1)];
    float denom = 2*stddev*stddev;
    for (int y = -radius; y<=radius; y++){
	for (int x = -radius; x<=radius; x++){
	    kernel[(y+radius)*(2*radius+1)+x+radius] = exp(-(x*x+y*y)/denom)/denom/M_PI;
	}
    }
    float3* temp = new float3[h_height*h_width];
    for (int i = 0; i<h_width*h_height; i++){
	float r,g,b;
	r=g=b=0;
	if (in[i].x > 1) r = in[i].x - 1;
	if (in[i].y > 1) g = in[i].y - 1;
	if (in[i].z > 1) b = in[i].z - 1;
	temp[i] = make_float3(r,g,b);
    }
    for (int y = 0; y<h_height; y++){
	for(int x = 0; x<h_width; x++){
	    float3 c = make_float3(0);
	    for (int ky = -radius; ky<=radius; ky++){
		for (int kx = -radius; kx<=radius; kx++){
		    if ((y+ky)>=0 && (y+ky)<h_height &&
			(x+kx)>=0 && (x+kx)<h_width){
			c = c+temp[(y+ky)*h_width+(x+kx)]*kernel[(ky+radius)*(2*radius+1)+kx+radius];
		    }
		}
	    }
	    out[y*h_width+x] = c/20 + in[y*h_width + x];
	}
    }

    delete[] temp;
}

int main(){
    // time the various parts of computation
    std::chrono::time_point<std::chrono::system_clock> t0,t1,t2,t3;

    t0 = std::chrono::system_clock::now();
    init_scene();

    float3* output_h = new float3[h_width*h_height];
    float3* output_d;

    cudaMalloc(&output_d, h_width*h_height*sizeof(float3));
    
    dim3 block(8,8,1);
    dim3 grid(h_width/block.x, h_height/block.y, 1);

    std::cout << "CUDA initialized" << std::endl;
    std::cout << "Start rendering..." << std::endl;

    t1 = std::chrono::system_clock::now();
    
    render_kernel<<<grid, block>>>(output_d);
    cudaDeviceSynchronize();
	
    cudaMemcpy(output_h, output_d, h_width*h_height*sizeof(float3),
	       cudaMemcpyDeviceToHost);
    cudaFree(output_d);

    t2 = std::chrono::system_clock::now();

    std::cout << "Done" << std::endl;

    float3* out = new float3[h_width*h_height];
    bloom(output_h, out, bloom_rad, 3);

    std::vector<unsigned char> image(h_width*h_height*4);
    for(int y = 0; y<h_height; y++){
	for(int x = 0; x<h_width; x++){
	    int idx = (y*h_width + x);
	    image[4*idx + 0] = to_int(out[idx].x);
	    image[4*idx + 1] = to_int(out[idx].y);
	    image[4*idx + 2] = to_int(out[idx].z);
	    image[4*idx + 3] = 255;
	}
    }
    lodepng::encode("test.png", image, h_width, h_height); // save image

    std::cout << "Saved image" << std::endl;
    
    delete[] output_h;

    t3 = std::chrono::system_clock::now();

    std::chrono::duration<double> init_time = t1 - t0;
    std::chrono::duration<double> kernel_time = t2 - t1;
    std::chrono::duration<double> post_time = t3 - t2;

    double is = init_time.count();
    double ks = kernel_time.count();
    double ps = post_time.count();
    int im = is/60;
    int km = ks/60;
    int pm = ps/60;
    is -= 60*im;
    ks -= 60*km;
    ps -= 60*pm;
    
    std::cout << std::endl;
    std::cout << "Initialization time: " << im << "m" << is << "s" << std::endl;
    std::cout << "Render time: " << km << "m" << ks << "s" << std::endl;
    std::cout << "Post process time: " << pm << "m" << ps << "s" << std::endl;
    
    return 0;
}
