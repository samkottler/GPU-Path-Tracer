//light types for dispersion
enum light_type {RED, GREEN, BLUE};

// light ray
struct Ray{
    float3 origin;
    float3 direction;
    float3 attenuation; //based on media light is moving through
    light_type light;
    __device__ Ray(float3 o, float3 d, float3 c, light_type l) : origin(o), direction(d), attenuation(c), light(l){}
};

//what a ray does when it hits an object
enum scatter_type {DIFF, SPEC, REFR, METAL, COAT, VOLUME};
//how to determine color of hit object
enum texture_type {CONST, CHECK, PIX};

// constant texture
struct const_text{
    float3 color;
    float3 emission;
    float density;
};

// checkered texture
struct check_text{
    float3 color1;
    float3 color2;
    float3 scale;
    float3 emission;
};

// texture from image
struct pix_text{
    int width;
    int height;
    int offset;
    __device__ float3 color_at(float2 uv){
	int x = width*uv.x;
	int y = height*uv.y;
	int idx = offset + y*width + x;
	return textures[idx];
    }
};

// struct to keep info about collisions organized
struct Hit_data{
    float3 p;
    float t;
    float3 normal;
    float2 uv;
    scatter_type scat;
    texture_type text;
    int texture_num;
    float dist;
    float shift;
};  


struct Sphere{
    float3 center;
    float radius;
    scatter_type scat;
    texture_type text;
    int texture_num;

    // use quadratic formula
    // if one positive solution, started inside sphere
    // difference of solutions in the distance between to two sides of the sphere
    inline __device__ float intersect(const Ray& r, float& dist) const{
	float3 to_origin = center - r.origin;
	float b = dot(to_origin, r.direction);
	float c = dot(to_origin,to_origin) - radius*radius;
	float disc = b*b-c;
	if(disc>0){
	    disc = sqrtf(disc);
	    float t = b - disc;
	    if (t>0.0001){
		dist = 1e20;
		return t;
	    }
	    t = b + disc;
	    if (t>0.0001){
		dist = 2*disc;
		return t;
	    }
	}
	return 0;
    }
};


// axis aligned bounding box
// I use this for boxes in the scene but really this should just be for the bvh
// needs to be changed in future
struct Box{
    float3 max;
    float3 min;
    scatter_type scat;
    texture_type text;
    int texture_num;
    
    __device__ float intersect(const Ray& r) const{
	if (min.x < r.origin.x && r.origin.x < max.x &&
	    min.y < r.origin.y && r.origin.y < max.y &&
	    min.z < r.origin.z && r.origin.z < max.z) return -1;
	
	float3 tmin = (min - r.origin) / r.direction;
	float3 tmax = (max - r.origin) / r.direction;

	float3 rmin = minf3(tmin,tmax);
	float3 rmax = maxf3(tmin,tmax);

	float minmax = minf(minf(rmax.x, rmax.y), rmax.z);
	float maxmin = maxf(maxf(rmin.x, rmin.y), rmin.z);

	if(minmax >= maxmin) return maxmin > 0.000001 ? maxmin : 0;
	else return 0;
    }

    // normal and uv only for using as actual object
    // remove in future
    __device__ float3 normal_at(const float3& p){
	float3 normal;
	if (fabs(min.x - p.x) < 0.0001) normal = make_float3(-1,0,0);
	else if (fabs(min.y - p.y) < 0.0001) normal = make_float3(0,-1,0);
	else if (fabs(min.z - p.z) < 0.0001) normal = make_float3(0,0,-1);
	else if (fabs(max.x - p.x) < 0.0001) normal = make_float3(1,0,0);
	else if (fabs(max.y - p.y) < 0.0001) normal = make_float3(0,1,0);
	else normal = make_float3(0,0,1);
	return normal;
    }
    __device__ float2 uv_at(const float3& p){
	float u = (max.x - p.x)/(min.x - max.x);
	float v = (max.y - p.y)/(min.y - max.y);
	float w = (max.z - p.z)/(min.z - max.z);
	float2 uv;
	if (fabs(min.x - p.x) < 0.0001) uv = make_float2(v,w);
	else if (fabs(min.y - p.y) < 0.0001) uv = make_float2(w,u);
	else if (fabs(min.z - p.z) < 0.0001) uv = make_float2(u,v);
	else if (fabs(max.x - p.x) < 0.0001) uv = make_float2(v,w);
	else if (fabs(max.y - p.y) < 0.0001) uv = make_float2(w,u);
	else uv = make_float2(u,v);
	return uv;
    }
};

// data for a single triangle
// currently has sepparate scatter and texture info for each triangle
// scatter and texture type and texture num should eventually be moved to mesh
struct Triangle{
    float3 p;
    float3 v0;
    float3 v1;
    float3 normal; //precompute and store. may not be faster needs testing
    scatter_type scat;
    texture_type text;
    int texture_num;
    float2 uv[3];
    int mesh_num;

    // use moller-trumbore algorithm to intersect quickly
    __device__ float intersect(const Ray& r) const{
	float3 tvec = r.origin - p;
	float3 pvec = cross(r.direction, v1);
	float det = dot(v0, pvec);
        
	det = __fdividef(1.0, det);

	float u = dot(tvec, pvec)*det;
	if (u < 0 || u > 1)
	    return -1e20;

	float3 qvec = cross(tvec, v0);

	float v = dot(r.direction, qvec) * det;

	if (v < 0 || (u+v) > 1)
	    return -1e20;

	return dot(v1, qvec) * det;
    }

    //turn into barycentric coorinates and calculate linear combination for uv
    __device__ float2 uv_at(float3 point){
	float3 v2 =  point-p;
	float d00 = dot(v0, v0);
	float d01 = dot(v0, v1);
	float d11 = dot(v1, v1);
	float d20 = dot(v2, v0);
	float d21 = dot(v2, v1);
	float denom = d00*d11 - d01*d01;
	float v = (d11*d20 - d01*d21)/denom;
	float w = (d00*d21 - d01*d20)/denom;
	float u = 1 - v - w;
	//printf("%f %f %f\n", u,v,w);
	float x = u*uv[0].x + v*uv[1].x + w*uv[2].x;
	float y = u*uv[0].y + v*uv[1].y + w*uv[2].y;
	//printf("(%f %f %f) %f %f\n", u,v,w,x,y);
	//printf("(%f %f %f) (%f %f %f) (%f %f %f) (%f %f %f) %f %f %f\n",
	//  p.x, p.y, p.z, (p+v0).x, (p+v0).y, (p+v0).z,
	//  (p+v1).x, (p+v1).y, (p+v1).z, point.x, point.y, point.z, u, v, w);
	return make_float2(x,y);
    }
};

//lists of geometry and textures on host
std::vector<const_text> h_const;
std::vector<check_text> h_check;
std::vector<float3> h_textures;
std::vector<pix_text> h_pix;
std::vector<Sphere> h_spheres;
std::vector<Box> h_boxes;
std::vector<Triangle> h_triangles;
