// STD include
#include <vector>
#include <thread>

// CUDA include
#ifdef __CUDACC__
#include "device_launch_parameters.h"
#include <cuda.h>
#include <cuda_runtime_api.h>
#include "cuda_runtime.h"
#endif

// OPENGL include
#include <GL/glut.h>
#include <GL/freeglut.h>

// STIM include
#include <stim/visualization/gl_network.h>
#include <stim/visualization/gl_aaboundingbox.h>
#include <stim/parser/arguments.h>
#include <stim/visualization/camera.h>
#include <stim/biomodels/flow.h>
#include <stim/visualization/colormap.h>
#include <stim/math/matrix.h>
#include <stim/grids/image_stack.h>
#include <stim/cuda/cudatools/error.h>
#include <stim/ui/progressbar.h>


//****************************parameter setting*********************************
// user input parameters
float u = 0.0f;					// viscosity
float h = 0.0f;					// height of edge(channel)
float dx, dy, dz;				// x, y and z image scaling(units/pixel)
float main_feeder_radii;		// default radii of main feeder (50um will be great for microfluidics manufacture)
float default_radii = 5.0f;		// default radii of network vertex
float minimum_radii = 0.0f;		// minimum radii that current machine can manufacture
float max_pressure = 0.0f;		// maximum pressure that the channel can bear
int X;							// workspace X
int Y;							// workspace Y
int size_x, size_y, size_z;		// size of image stack
std::string units;				// units
std::string stackdir = "";		// directory where image stacks will be stored

// window console parameters
int mouse_x = -5;				// mouse x window position
int mouse_y = -5;				// mouse y window position
int vX;							// viewport X
int vY;							// viewport Y

// hard-coded parameters
float delta = 0.01f;			// discrepancy
float eps = 15.0f;				// epsilon threshold
std::vector<std::string> menu_option = { "generate network", "simulation", "build inlet", "build outlet", "manufacture" };
int cur_menu_num;				// number of current menu option
int new_menu_num;				// number of new menu option
int mode;						// menu options
int mods;						// special keyboard input
float border = 20.0f;			// bar edge position
float radii_factor = 0.4f;		// change ratio of network vertex radii
GLint subdivision = 20;			// slices and stacks
float cur_max_radii = 0.0f;		// store the maximum radii in the network for manufacture

// new structure type definition
struct vertex {
	stim::vec3<float> c;		// coordinates
	float r = default_radii;	// radii
};
struct edge {
	unsigned p[2];				// start and end vertex indices
	float v = 0.0f;				// velocity along edge
};
struct sphere {
	stim::vec3<float> c;		// center of sphere
	float r;					// radii
};
struct cylinder {				// radii changes gradually
	stim::vec3<float> c1;		// center of geometry start hat
	stim::vec3<float> c2;		// center of geometry end hat
	float r1;					// radii at start hat
	float r2;					// radii at end hat
};

// parameters for generating new networks
bool generate_network = false;	// flag indicates in generating network mode
bool first_click = true;		// flag indicates first click of one line of edges
bool flag = false;				// flag indicates found a near vertex or edge
unsigned num = 0;				// number of vertex in a new line
unsigned iter = 0;				// iterator indicates index of current vertex
unsigned name = 0;				// output network's main name in sequences
	unsigned sub_name = 0;		// output network's sub_name in sequences
vertex new_vertex;				// stores current acceptable vertex
vertex tmp_vertex;				// temporarily stores a vertex when moving mouse
edge new_edge;					// stores current acceptable edge
edge tmp_edge;					// temporarily stores a edge when moving mouse
std::vector<unsigned> dangle_vertex;	// boundary(dangle) vertices list
stim::vec3<float> L = stim::vec3<float>(FLT_MAX, FLT_MAX, FLT_MAX);		// minimum point in the bounding box
stim::vec3<float> U = stim::vec3<float>(-FLT_MAX, -FLT_MAX, -FLT_MAX);	// maximum point in the bounding box
std::vector<unsigned> color_scheme;		// color scheme for each edge
unsigned color_index = 0;

// parameters for simulation
bool simulation = false;			// flag indicates in simulation network mode
bool first_simulation = true;		// initialize simulation, all inlet to maximum pressure, all outlet to zero pressure
bool select_pressure = false;		// flag indicates having selected a vertex to modify pressure, next step is to set specific pressure value	
bool select_radii = false;			// flag indicates having selected a vertex to change radii, next step is to set specific radii value
bool radii_changed = false;			// flag indicates one vertex has been changed radii
bool grow = false;					// flag indicates grow new line of edges
unsigned pressure_index = 0;		// index of picked vertex for pressure
unsigned radii_index = 0;			// index of picked vertex for radii
unsigned edge_index = 0;			// index of picked edge
float max_v;						// maximum velocity in units / s
float min_v;
stim::flow<float> Flow;				// flow object for calculating network fluid flow
std::vector<typename stim::triple<unsigned, unsigned, float> > input;	// first one store which vertex, second one stores which edge, third one stores in/out volume flow rate of that vertex
std::vector<typename stim::triple<unsigned, unsigned, float> > output;
std::vector<unsigned char> color;	// color map based on velocity
bool color_bound = false;			// flag indicates color map has been bound to 1D texture
std::vector<int> velocity_map;		// velocity map
std::vector<typename edge> tmp_E;	// temp list of edges
unsigned new_num = 0;				// number of new growing vertex

// parameters for building bridge
bool build_inlet = false;			// flag indicates in building inlet mode
bool build_outlet = false;			// flag indicates in building outlet mode
bool select_bridge = false;			// flag indicates now user can select bridge to modify
bool select_corner = false;			// flag indicates having selected a bridge to modify, the next click is to choose a new position for the corner vertex
bool inlet_done = false;			// finished choosing the inlet main feeder position
bool outlet_done = false;			// finished choosing the outlet main feeder position
std::vector<typename stim::bridge<float> > inlet;				// input bridge
std::vector<typename stim::bridge<float> > outlet;				// output bridge
stim::vec3<float> inlet_port;									// inlet main feeder port
stim::vec3<float> outlet_port;									// outlet main feeder port
stim::vec3<float> corner_vertex;								// corner vertex
unsigned bridge_index;											// selected bridge index
float inlet_flow_rate = 0.0f;		// volume flow rate at main inlet feeder
float outlet_flow_rate = 0.0f;		// volume flow rate at main outlet feeder
float inlet_pressure;				// pressure at main inlet feeder
float outlet_pressure;				// pressure at main outlet feeder
unsigned min_input_index;			// maximum output pressure index
unsigned max_output_index;			// minimum input pressure index		
std::vector<bool> inlet_feasibility;// list of flag indicates ith inlet bridge feasibility
std::vector<bool> outlet_feasibility;

// parameters for manufacture
bool manufacture = false;			// flag indicates in manufacture mode
bool mask_done = false;				// flag indicates having made a mask

// network
unsigned num_edge = 0;				// number of edges in current network
unsigned num_vertex = 0;			// number of vertices in current network
std::vector<vertex> V;				// list of vertices
std::vector<edge> E;				// list of edges

// image stack
stim::image_stack<unsigned char, float> I;		// image stack object
std::vector<sphere> A;							// sphere model for making image stack
	unsigned feeder_start_index;
std::vector<cylinder> B;						// cylinder model for making image stack
	unsigned bridge_start_index;

// camera object
stim::camera cam;					// camera object
float camera_factor = 1.2f;			// start point of the camera as a function of X and Y size

// colors
#define JACK_CTRL_PTS 11
static float JACKCP[JACK_CTRL_PTS * 3] = { 0.671f, 0.851f, 0.914f,
										0.502f, 0.804f, 0.757f,
										0.651f, 0.851f, 0.416f,
										0.945f, 0.714f, 0.855f,
										0.600f, 0.439f, 0.671f,
										0.914f, 0.761f, 0.490f,
										0.729f, 0.729f, 0.729f,
										0.957f, 0.647f, 0.510f,
										0.996f, 0.878f, 0.565f,
										0.992f, 0.722f, 0.388f,
										0.957f, 0.427f, 0.263f };


//****************************auxiliary functions*********************************
// find the nearest vertex of current click position
// return true and a value if found
inline bool epsilon_vertex(int x, int y, unsigned& v) {
	
	float d = FLT_MAX;									// minimum distance between 2 vertices
	float tmp_d = 0.0f;									// temporary stores distance for loop
	unsigned tmp_i = 0;									// temporary stores connection index for loop
	stim::vec3<float> tmp_v;							// temporary stores current loop point
	d = FLT_MAX;										// set to max of float number
	
	for (unsigned i = 0; i < V.size(); i++) {
		tmp_v = stim::vec3<float>((float)x, (float)(vY - y), 0.0f);
		tmp_v[0] = tmp_v[0] * (float)X / vX;
		tmp_v[1] = tmp_v[1] * (float)Y / vY;

		tmp_v = tmp_v - V[i].c;							// calculate a vector between two vertices
		tmp_d = tmp_v.len();							// calculate length of that vector
		if (tmp_d < d) {
			d = tmp_d;									// if found a nearer vertex 
			tmp_i = i;									// get the index of that vertex
		}
	}
	if (d < eps) {										// if current click is close to vertex we set before
		// must have at least three point to make a plane or loop 
		if (tmp_i < num && (tmp_i == V.size() - 1 || tmp_i == V.size() - 2) && !first_click && mods == 0) {
			Sleep(100);							// make flash effect
			std::cout << "\r";
			std::cout << "[  ERROR  ]  ";
			std::cout << "You can't do that!";
			std::cout.flush();
		}
		else {
			v = tmp_i;									// copy the extant vertex's index to v
		}
		return true;
	}

	return false;
}

// check out whether the projection of v0 onto line segment v1-v2 is on extensed line
// set distance to FLT_MAX if true
inline void is_outside(stim::vec3<float> v0, stim::vec3<float> v1, stim::vec3<float> v2, float &distance) {
	float a = (v0 - v1).dot((v2 - v1).norm());
	float b = (v0 - v2).dot((v1 - v2).norm());
	float length = (v2 - v1).len();
	if (a > length || b > length)
		distance = FLT_MAX;
}

// find the nearest inlet/outlet connection line of current click position
// return true and a value if found
inline bool epsilon_edge(int x, int y, unsigned &idx) {
	
	float d = FLT_MAX;
	float tmp_d;
	unsigned tmp_i;
	stim::vec3<float> v1;
	stim::vec3<float> v2;
	stim::vec3<float> v0 = stim::vec3<float>((float)x, (float)(vY - y), 0.0f);
	v0[0] = v0[0] * (float)X / vX;
	v0[1] = v0[1] * (float)Y / vY;

	if (build_inlet) {
		for (unsigned i = 0; i < inlet.size(); i++) {
			if (inlet[i].V.size() == 2) {		// direct line connection
				v1 = inlet[i].V[0];				// the inlet port vertex
				v2 = inlet[i].V[1];				// the dangle vertex

				// the distance between a point and a line segment, d = (|(x0 - x1)x(x0 - x2)|) / (|x2 - x1|)
				tmp_d = ((v0 - v1).cross(v0 - v2)).len() / (v2 - v1).len();
				if (tmp_d < d) {
					d = tmp_d;
					tmp_i = i;
					// check whether the projection is on the line segment
					is_outside(v0, v1, v2, d);
				}
			}
			else if (inlet[i].V.size() == 3) {	// broken line connection
				// first half of bridge
				v1 = inlet[i].V[0];				// the inlet port vertex
				v2 = inlet[i].V[1];				// the corner vertex

				// the distance between a point and a line segment, d = (|(x0 - x1)x(x0 - x2)|) / (|x2 - x1|)
				tmp_d = ((v0 - v1).cross(v0 - v2)).len() / (v2 - v1).len();
				if (tmp_d < d) {
					d = tmp_d;
					tmp_i = i;
					is_outside(v0, v1, v2, d);
				}
			
				// second half of bridge
				v1 = inlet[i].V[1];				// the corner vertex
				v2 = inlet[i].V[2];				// the dangle vertex

				// the distance between a point and a line segment, d = (|(x0 - x1)x(x0 - x2)|) / (|x2 - x1|)
				tmp_d = ((v0 - v1).cross(v0 - v2)).len() / (v2 - v1).len();
				if (tmp_d < d) {
					d = tmp_d;
					tmp_i = i;
					is_outside(v0, v1, v2, d);
				}	
			}	
		}

		if (d < eps) {
			idx = tmp_i;
			return true;
		}
	}
	else if (build_outlet) {
		for (unsigned i = 0; i < outlet.size(); i++) {
			if (outlet[i].V.size() == 2) {		// direct line connection
				// first half of bridge
				v1 = outlet[i].V[0];			// the inlet port vertex
				v2 = outlet[i].V[1];			// the dangle vertex

				// the distance between a point and a line segment, d = (|(x0 - x1)x(x0 - x2)|) / (|x2 - x1|)
				tmp_d = ((v0 - v1).cross(v0 - v2)).len() / (v2 - v1).len();
				if (tmp_d < d) {
					d = tmp_d;
					tmp_i = i;
					is_outside(v0, v1, v2, d);
				}	
			}
			else if (outlet[i].V.size() == 3) {	// broken line connection
				v1 = outlet[i].V[0];			// the inlet port vertex
				v2 = outlet[i].V[1];			// the corner vertex

				// the distance between a point and a line segment, d = (|(x0 - x1)x(x0 - x2)|) / (|x2 - x1|)
				tmp_d = ((v0 - v1).cross(v0 - v2)).len() / (v2 - v1).len();
				if (tmp_d < d) {
					d = tmp_d;
					tmp_i = i;
					is_outside(v0, v1, v2, d);
				}
			
				// second half of bridge
				v1 = outlet[i].V[1];			// the corner vertex
				v2 = outlet[i].V[2];			// the dangle vertex

				// the distance between a point and a line segment, d = (|(x0 - x1)x(x0 - x2)|) / (|x2 - x1|)
				tmp_d = ((v0 - v1).cross(v0 - v2)).len() / (v2 - v1).len();
				if (tmp_d < d) {
					d = tmp_d;
					tmp_i = i;
					is_outside(v0, v1, v2, d);
				}
			}
		}

		if (d < eps) {							// check to see whether the smallest distance is within the threshold
			idx = tmp_i;
			return true;
		}
	}
	return false;
}

// find the nearest edge
// retrun true, edge index and index within edge if found
inline bool epsilon_edge(int x, int y, unsigned &idx, unsigned &i) {

	float d = FLT_MAX;
	float tmp_d;
	unsigned tmp_i;
	stim::vec3<float> v1;
	stim::vec3<float> v2;
	stim::vec3<float> v0 = stim::vec3<float>((float)x, (float)(vY - y), 0.0f);
	v0[0] = v0[0] * (float)X / vX;
	v0[1] = v0[1] * (float)Y / vY;

	for (unsigned i = 0; i < E.size(); i++) {
		v1 = V[E[i].p[0]].c;			// starting vertex
		v2 = V[E[i].p[1]].c;			// ending vertex

		// the distance between a point and a line segment, d = (|(x0 - x1)x(x0 - x2)|) / (|x2 - x1|)
		tmp_d = ((v0 - v1).cross(v0 - v2)).len() / (v2 - v1).len();
		if (tmp_d < d) {
			d = tmp_d;
			tmp_i = i;
			// check whether the projection is on the line segment
			is_outside(v0, v1, v2, d);
		}
	}

	if (d < eps) {
		idx = tmp_i;					// get the edge index
		float px;
		float py;

		// get the projection coordinates
		v1 = V[E[idx].p[0]].c;
		v2 = V[E[idx].p[1]].c;
		float dx = v2[0] - v1[0];
		float dy = v2[1] - v1[1];
		float dAB = dx * dx + dy * dy;

		float u = ((v0[0] - v1[0]) * dx + (v0[1] - v1[1]) * dy) / dAB;
		px = v1[0] + u * dx;
		py = v1[1] + u * dy;

		float l = (v1 - v2).len();
		tmp_d = sqrt(std::pow(px - v1[0], 2) + std::pow(py - v1[1], 2));
		if (tmp_d < l - tmp_d) 			// if the projection is near starting vertex
			i = 0;
		else
			i = 1;
		return true;
	}

	return false;
}

// check whether there is a edge between two vertices
// return true if found
inline bool is_edge(unsigned idx) {
	
	for (unsigned i = 0; i < E.size(); i++) {	// brute force method
		if (E[i].p[0] == new_edge.p[0] && E[i].p[1] == idx)
			return true;
		else if (E[i].p[1] == new_edge.p[0] && E[i].p[0] == idx)
			return true;
	}

	return false;
}

// find the distance between two vertices
inline float length(unsigned i) {
	stim::vec3<float> v1 = V[E[i].p[0]].c;
	stim::vec3<float> v2 = V[E[i].p[1]].c;
	
	v1 = v1 - v2;

	return v1.len();
}

// find the average radius of one edge
inline float radius(unsigned i) {
	return (V[E[i].p[0]].r + V[E[i].p[1]].r) / 2;
}

// find two envelope caps for two spheres
// @param cp1, cp2: list of points on the cap
// @param center1, center2: center point of cap
// @param r1, r2: radii of cap
inline void find_envelope(std::vector<typename stim::vec3<float> > &cp1, std::vector<typename stim::vec3<float> > &cp2, stim::vec3<float> center1, stim::vec3<float> center2, float r1, float r2) {

	stim::vec3<float> tmp_d;
	if (r1 == r2) {						// two vertices have the same radius
		tmp_d = center2 - center1;		// calculate the direction vector
		tmp_d = tmp_d.norm();
		stim::circle<float> tmp_c;		// in order to get zero direction vector
		tmp_c.rotate(tmp_d);

		stim::circle<float> c1(center1, r1, tmp_d, tmp_c.U);
		stim::circle<float> c2(center2, r2, tmp_d, tmp_c.U);
		cp1 = c1.glpoints(subdivision);
		cp2 = c2.glpoints(subdivision);
	}
	else {
		if (r1 < r2) {					// switch index, we always want r1 to be larger than r2
			stim::vec3<float> tmp_c = center2;
			center2 = center1;
			center1 = tmp_c;
			float tmp_r = r2;
			r2 = r1;
			r1 = tmp_r;
		}
		tmp_d = center2 - center1;		// bigger one points to smaller one
		tmp_d = tmp_d.norm();

		float D = (center1 - center2).len();
		stim::vec3<float> exp;
		exp[0] = (center2[0] * r1 - center1[0] * r2) / (r1 - r2);
		exp[1] = (center2[1] * r1 - center1[1] * r2) / (r1 - r2);

		stim::vec3<float> t1, t2, t3, t4;
		t1[2] = t2[2] = t3[2] = t4[2] = 0.0f;

		// first two
		t1[0] = pow(r1, 2)*(exp[0] - center1[0]);
		t1[0] += r1*(exp[1] - center1[1])*sqrt(pow((exp[0] - center1[0]), 2) + pow((exp[1] - center1[1]), 2) - pow(r1, 2));
		t1[0] /= (pow((exp[0] - center1[0]), 2) + pow((exp[1] - center1[1]), 2));
		t1[0] += center1[0];

		t2[0] = pow(r1, 2)*(exp[0] - center1[0]);
		t2[0] -= r1*(exp[1] - center1[1])*sqrt(pow((exp[0] - center1[0]), 2) + pow((exp[1] - center1[1]), 2) - pow(r1, 2));
		t2[0] /= (pow((exp[0] - center1[0]), 2) + pow((exp[1] - center1[1]), 2));
		t2[0] += center1[0];

		t1[1] = pow(r1, 2)*(exp[1] - center1[1]);
		t1[1] -= r1*(exp[0] - center1[0])*sqrt(pow((exp[0] - center1[0]), 2) + pow((exp[1] - center1[1]), 2) - pow(r1, 2));
		t1[1] /= (pow((exp[0] - center1[0]), 2) + pow((exp[1] - center1[1]), 2));
		t1[1] += center1[1];

		t2[1] = pow(r1, 2)*(exp[1] - center1[1]);
		t2[1] += r1*(exp[0] - center1[0])*sqrt(pow((exp[0] - center1[0]), 2) + pow((exp[1] - center1[1]), 2) - pow(r1, 2));
		t2[1] /= (pow((exp[0] - center1[0]), 2) + pow((exp[1] - center1[1]), 2));
		t2[1] += center1[1];

		// check the correctness of the points
		//float s = (center1[1] - t1[1])*(exp[1] - t1[1]) / ((t1[0] - center1[0])*(t1[0] - exp[0]));
		//if (s != 1) {			// swap t1[1] and t2[1]
		//	float tmp_t = t2[1];
		//	t2[1] = t1[1];
		//	t1[1] = tmp_t;
		//}

		// second two
		t3[0] = pow(r2, 2)*(exp[0] - center2[0]);
		t3[0] += r2*(exp[1] - center2[1])*sqrt(pow((exp[0] - center2[0]), 2) + pow((exp[1] - center2[1]), 2) - pow(r2, 2));
		t3[0] /= (pow((exp[0] - center2[0]), 2) + pow((exp[1] - center2[1]), 2));
		t3[0] += center2[0];

		t4[0] = pow(r2, 2)*(exp[0] - center2[0]);
		t4[0] -= r2*(exp[1] - center2[1])*sqrt(pow((exp[0] - center2[0]), 2) + pow((exp[1] - center2[1]), 2) - pow(r2, 2));
		t4[0] /= (pow((exp[0] - center2[0]), 2) + pow((exp[1] - center2[1]), 2));
		t4[0] += center2[0];

		t3[1] = pow(r2, 2)*(exp[1] - center2[1]);
		t3[1] -= r2*(exp[0] - center2[0])*sqrt(pow((exp[0] - center2[0]), 2) + pow((exp[1] - center2[1]), 2) - pow(r2, 2));
		t3[1] /= (pow((exp[0] - center2[0]), 2) + pow((exp[1] - center2[1]), 2));
		t3[1] += center2[1];

		t4[1] = pow(r2, 2)*(exp[1] - center2[1]);
		t4[1] += r2*(exp[0] - center2[0])*sqrt(pow((exp[0] - center2[0]), 2) + pow((exp[1] - center2[1]), 2) - pow(r2, 2));
		t4[1] /= (pow((exp[0] - center2[0]), 2) + pow((exp[1] - center2[1]), 2));
		t4[1] += center2[1];

		// check the correctness of the points
		//s = (center2[1] - t3[1])*(exp[1] - t3[1]) / ((t3[0] - center2[0])*(t3[0] - exp[0]));
		//if (s != 1) {			// swap t1[1] and t2[1]
		//	float tmp_t = t4[1];
		//	t4[1] = t3[1];
		//	t3[1] = tmp_t;
		//}

		stim::vec3<float> d1;
		float dot;
		float a;
		float new_r;
		stim::vec3<float> new_u;
		stim::vec3<float> new_c;

		// calculate the bigger circle
		d1 = t1 - center1;
		dot = d1.dot(tmp_d);
		a = dot / (r1 * 1) * r1;			// a = cos(alpha) * radii
		new_c = center1 + a * tmp_d;
		new_r = sqrt(pow(r1, 2) - pow(a, 2));
		new_u = t1 - new_c;

		stim::circle<float> c1(new_c, new_r, tmp_d, new_u);
		cp1 = c1.glpoints(subdivision);

		// calculate the smaller circle
		d1 = t3 - center2;
		dot = d1.dot(tmp_d);
		a = dot / (r2 * 1) * r2;
		new_c = center2 + a * tmp_d;
		new_r = sqrt(pow(r2, 2) - pow(a, 2));
		new_u = t3 - new_c;

		stim::circle<float> c2(new_c, new_r, tmp_d, new_u);
		cp2 = c2.glpoints(subdivision);
	}
}

// check to see whether current bridge is acceptable
// if it is not acceptable, print error reminder
inline void is_acceptable() {
	
	if (build_inlet) {
		unsigned midx;						// get the index from inlet list
		for (unsigned i = 0; i < inlet.size(); i++) {
			if (inlet[i].v[0] == min_input_index) {
				midx = i;
				break;
			}
		}
		float tmp_r;
		unsigned idx;
		std::vector<bool> tmp(inlet.size(), true);
		std::swap(tmp, inlet_feasibility);
		for (unsigned i = 0; i < inlet.size(); i++) {
			idx = inlet[i].v[0];
			if (i != midx) {
				if (mode == 2)
					tmp_r = ((Flow.pressure[min_input_index] + ((12 * u * inlet[midx].l * inlet[midx].Q) / (std::pow(h, 3) * 2 * minimum_radii)) - Flow.pressure[idx]) * (std::pow(h, 3)) / (12 * u * inlet[i].l * inlet[i].Q)) / 2;
				else if (mode == 3)
					tmp_r = (Flow.pressure[min_input_index] + ((8 * u * inlet[midx].l * inlet[midx].Q) / (std::pow(minimum_radii, 4) * (float)stim::PI)) - Flow.pressure[idx]) * (float)stim::PI / (8 * u * inlet[i].l * inlet[i].Q);

				if (tmp_r <= 0) {			// degenerate case where radii ie less than zero
					Sleep(100);				// make flash effect
					std::cout << "\r";
					std::cout << "[  ERROR  ]  ";
					std::cout << "Inlet bridge for vertex " << min_input_index << " is not feasible";
					inlet_feasibility[i] = false;
					break;
				}
				else 						// feasible
					inlet_feasibility[i] = true;
			}
		}
	}
	else if (build_outlet) {
		unsigned midx;						// get the index from outlet list
		for (unsigned i = 0; i < outlet.size(); i++) {
			if (outlet[i].v[0] == max_output_index) {
				midx = i;
				break;
			}
		}
		float tmp_r;
		unsigned idx;
		std::vector<bool> tmp(outlet.size(), true);
		std::swap(tmp, outlet_feasibility);
		for (unsigned i = 0; i < outlet.size(); i++) {
			idx = outlet[i].v[0];
			if (i != midx) {
				if (mode == 2)
					tmp_r = ((Flow.pressure[idx] - (Flow.pressure[max_output_index] - (12 * u * outlet[midx].l * outlet[midx].Q) / (std::pow(h, 3) * 2 * minimum_radii))) * (std::pow(h, 3)) / (12 * u * outlet[i].l * outlet[i].Q)) / 2;
				else if (mode == 3)
					tmp_r = (Flow.pressure[idx] - (Flow.pressure[max_output_index] - (8 * u * outlet[midx].l * outlet[midx].Q) / (std::pow(minimum_radii, 4) * (float)stim::PI))) * (float)stim::PI / (8 * u * outlet[i].l * outlet[i].Q);
				if (tmp_r <= 0) {									// not enough length to satisfy to situation
					std::cout << "\r";
					std::cout << "[  ERROR  ]  ";
					std::cout << "Outlet bridge for vertex " << max_output_index << " is not feasible";
					outlet_feasibility[i] = false;
					break;
				}
				else						// feasible
					outlet_feasibility[i] = true;
			}
		}
	}
}

//****************************simulation functions*********************************
// get the network information
void get_background() {
	
	num_edge = E.size();						// get the number of edge on current network
	num_vertex = V.size();						// get the number of vertices on current network

	// get the bounding box of current network
	float tmp;
	for (unsigned i = 0; i < num_vertex; i++) {
		for (unsigned j = 0; j < 3; j++) {
			tmp = V[i].c[j];
			if (tmp < L[j])
				L[j] = tmp;
			if (tmp > U[j])
				U[j] = tmp;
		}
	}

	// get the dangle vertex
	dangle_vertex.clear();
	for (unsigned i = 0; i < num_vertex; i++) {
		unsigned k = 0;
		for (unsigned j = 0; j < num_edge; j++) {
			if (E[j].p[0] == i || E[j].p[1] == i)
				k++;
		}
		if (k == 1)
			dangle_vertex.push_back(i);
	}

	// print out
	std::cout << "OBJECT			NUMBER" << std::endl;
	std::cout << "edge			" << num_edge << std::endl;
	std::cout << "vertex			" << num_vertex << std::endl;
	std::cout << "dangle vertex		" << dangle_vertex.size() << std::endl;

	Flow.init(num_edge, num_vertex);	// initialize flow object
}

// initialize flow
void flow_initialize() {

	// clear up non-dangle vertex pressure
	for (unsigned i = 0; i < num_vertex; i++) {
		bool is_dangle = false;
		for (unsigned j = 0; j < dangle_vertex.size(); j++) {
			if (dangle_vertex[j] == i)
				is_dangle = true;
		}
		if (!is_dangle)
			Flow.P[i] = 0;
	}

	if (!grow) {					// when it is to grow a new edge, do not initialize again
		float mid = 0.0f;
		for (unsigned i = 0; i < dangle_vertex.size(); i++) {
			mid += V[dangle_vertex[i]].c[0];
		}
		mid /= dangle_vertex.size();

		for (unsigned i = 0; i < dangle_vertex.size(); i++) {
			if (V[dangle_vertex[i]].c[0] <= mid)
				Flow.P[dangle_vertex[i]] = max_pressure - i * delta;	// should set minor discrepancy
			else
				Flow.P[dangle_vertex[i]] = (i + 1) * delta;				// algorithm treat 0 as no initial pressure
		}
	}
}

// find the stable flow state
void find_stable_state(float threshold = 0.01f) {
	
	// clear up last time simulation
	input.clear();
	output.clear();
	std::vector<float> zero_QQ(num_vertex);
	std::swap(Flow.QQ, zero_QQ);
	std::vector<float> zero_pressure(num_vertex);
	std::swap(Flow.pressure, zero_pressure);

	// set the conductance matrix of flow object
	unsigned start_vertex = 0;
	unsigned end_vertex = 0;
	for (unsigned i = 0; i < num_edge; i++) {
		start_vertex = E[i].p[0];		// get the start vertex index of current edge
		end_vertex = E[i].p[1];			// get the end vertex index of current edge
		if (mode == 2) {
			Flow.C[start_vertex][end_vertex] = -(2 * radius(i) * std::pow(h, 3)) / (12 * u * length(i));		// UNITS: g/mm^4/s
		}
		else if (mode == 3) {
			Flow.C[start_vertex][end_vertex] = -((float)stim::PI * std::pow(radius(i), 4)) / (8 * u * length(i));
		}
		Flow.C[end_vertex][start_vertex] = Flow.C[start_vertex][end_vertex];
	}
	// set the diagonal to the negative sum of row element
	float sum = 0.0;
	for (unsigned i = 0; i < num_vertex; i++) {
		for (unsigned j = 0; j < num_vertex; j++) {
			sum += Flow.C[i][j];
		}
		Flow.C[i][i] = -sum;
		sum = 0.0;
	}

	// get the Q' vector QQ
	// matrix manipulation to zero out the conductance matrix as defined by the boundary values that were enterd
	for (unsigned i = 0; i < num_vertex; i++) {
		if (Flow.P[i] != 0) {			// for every dangle vertex
			for (unsigned j = 0; j < num_vertex; j++) {
				if (j == i) {
					Flow.QQ[i] = Flow.C[i][i] * Flow.P[i];
				}
				else {
					Flow.C[i][j] = 0;
					Flow.QQ[j] = Flow.QQ[j] - Flow.C[j][i] * Flow.P[i];
					Flow.C[j][i] = 0;
				}
			}
		}
	}

	// get the inverse of conductance matrix
	stim::matrix<float> _C(num_vertex, num_vertex);
	//float** _C = (float**)calloc(num_vertex, sizeof(float*));
	//for (unsigned i = 0; i < num_vertex; i++) {
	//	_C[i] = new float[num_vertex]();
	//}

	Flow.inversion(Flow.C, num_vertex, _C.data());

	// get the pressure in the network
	for (unsigned i = 0; i < num_vertex; i++) {
		for (unsigned j = 0; j < num_vertex; j++) {
			//Flow.pressure[i] += _C[i][j] * Flow.QQ[j];
			Flow.pressure[i] += _C(i, j) * Flow.QQ[j];
		}
	}

	// get the flow state from known pressure
	float start_pressure = 0.0;
	float end_pressure = 0.0;
	float deltaP = 0.0;
	for (unsigned i = 0; i < num_edge; i++) {
		start_vertex = E[i].p[0];
		end_vertex = E[i].p[1];
		start_pressure = Flow.pressure[start_vertex];		// get the start vertex pressure of current edge
		end_pressure = Flow.pressure[end_vertex];			// get the end vertex pressure of current edge
		deltaP = start_pressure - end_pressure;				// deltaP = Pa - Pb

		Flow.Q[i].first = start_vertex;
		Flow.Q[i].second = end_vertex;
		if (mode == 2) {
			Flow.Q[i].third = (2 * radius(i) * std::pow(h, 3) * deltaP) / (12 * u * length(i));
			E[i].v = Flow.Q[i].third / (h * 2 * radius(i));													
		}
		else if (mode == 3) {
			Flow.Q[i].third = ((float)stim::PI * std::pow(radius(i), 4) * deltaP) / (8 * u * length(i));		
			E[i].v = Flow.Q[i].third / ((float)stim::PI * std::pow(radius(i), 2));						
		}
	}

	// find both input and output vertex
	stim::triple<unsigned, unsigned, float> tmp;
	unsigned N = dangle_vertex.size();				// get the number of dangle vertex
	unsigned idx = 0;
	for (unsigned i = 0; i < N; i++) {				// for every boundary vertex
		idx = dangle_vertex[i];
		for (unsigned j = 0; j < num_edge; j++) {	// for every edge
			if (Flow.Q[j].first == idx) {			// starting vertex
				if (Flow.Q[j].third > 0) {			// flow comes in
					tmp.first = idx;
					tmp.second = j;
					tmp.third = Flow.Q[j].third;
					input.push_back(tmp);
					break;
				}
				// their might be a degenerate case that it equals to 0?
				else if (Flow.Q[j].third < 0) {		// flow comes out
					tmp.first = idx;
					tmp.second = j;
					tmp.third = -Flow.Q[j].third;
					output.push_back(tmp);
					break;
				}
			}
			else if (Flow.Q[j].second == idx) {		// ending vertex
				if (Flow.Q[j].third > 0) {			// flow comes in
					tmp.first = idx;
					tmp.second = j;
					tmp.third = Flow.Q[j].third;
					output.push_back(tmp);
					break;
				}
				// their might be a degenerate case that it equals to 0?
				else if (Flow.Q[j].third < 0) {		// flow comes out
					tmp.first = idx;
					tmp.second = j;
					tmp.third = -Flow.Q[j].third;
					input.push_back(tmp);
					break;
				}
			}
		}
	}
	
	// find the absolute maximum velocity and minimum velocity
	std::vector<float> abs_V(num_edge);
	for (unsigned i = 0; i < num_edge; i++) {
		abs_V[i] = std::fabsf(E[i].v);
		if (abs_V[i] < threshold)
			abs_V[i] = 0.0f;
	}

	max_v = *std::max_element(abs_V.begin(), abs_V.end());
	min_v = *std::min_element(abs_V.begin(), abs_V.end());

	// get the color map based on velocity range along the network
	color.clear();
	if (dangle_vertex.size() == 2 && num_edge - num_vertex + 1 <= 0) 		// only one inlet and one outlet
		color.resize(num_edge * 3, (unsigned char)0);
	else {
		color.resize(num_edge * 3);
		stim::cpu2cpu<float>(&abs_V[0], &color[0], num_edge, min_v, max_v, stim::cmBrewer);
	}
	color_bound = true;

	// sort the velocity bar in ascending order
	velocity_map.resize(num_edge);
	for (unsigned i = 0; i < num_edge; i++)
		velocity_map[i] = i;
	std::sort(velocity_map.begin(), velocity_map.end(), [&](int x, int y) {return abs_V[x] < abs_V[y]; });

	Flow.reset(num_vertex);				// reset flow object for next time simulation

	// find the minimum pressure input port
	if (input.size()) {
		min_input_index = input[0].first;
		for (unsigned i = 1; i < input.size(); i++) {
			unsigned idx = input[i].first;
			if (Flow.pressure[idx] < Flow.pressure[min_input_index])
				min_input_index = idx;
		}
	}
	// find the minimum pressure output port
	if (output.size()) {
		max_output_index = output[0].first;
		for (unsigned i = 1; i < output.size(); i++) {
			unsigned idx = output[i].first;
			if (Flow.pressure[idx] > Flow.pressure[max_output_index])
				max_output_index = idx;
		}
	}
	
	// get the number of input/output
	inlet_feasibility.resize(input.size(), true);
	outlet_feasibility.resize(output.size(), true);
}

// display and output final state
void show_stable_state() {
	
	std::cout << std::endl;
	// save the pressure information to CSV file
	std::string p_filename = "pressure.csv";
	std::ofstream p_file;
	p_file.open(p_filename.c_str());
	p_file << "Vertex, Pressure(g/" << units << "/s^2)" << std::endl;
	for (unsigned i = 0; i < num_vertex; i++)
		p_file << i << "," << Flow.pressure[i] << std::endl;
	p_file.close();
	// show the pressure information in console box
	std::cout << "PRESSURE(g/" << units << "/s^2):" << std::endl;
	for (unsigned i = 0; i < num_vertex; i++) {
		std::cout << "[" << i << "] " << Flow.pressure[i] << std::endl;
	}

	// save the flow information to CSV file
	std::string f_filename = "flow.csv";
	std::ofstream f_file;
	f_file.open(f_filename.c_str());
	f_file << "Edge, Volume flow rate(" << units << "^3/s)" << std::endl;
	for (unsigned i = 0; i < num_edge; i++)
		f_file << Flow.Q[i].first << "->" << Flow.Q[i].second << "," << Flow.Q[i].third << std::endl;
	f_file.close();

	// show the flow rate information in console box
	std::cout << "VOLUME FLOW RATE(" << units << "^3/s):" << std::endl;
	for (unsigned i = 0; i < num_edge; i++) {
		std::cout << "(" << Flow.Q[i].first << "," << Flow.Q[i].second << ")" << Flow.Q[i].third << std::endl;
	}
}


//****************************manufacture functions*********************************
// indicator functions
// indicator for sphere
__global__ void find_near_sphere(const sphere* V, unsigned num, size_t* R, float* S, unsigned char* ptr, unsigned z, int Size) {

	unsigned ix = blockDim.x * blockIdx.x + threadIdx.x;	// col
	unsigned iy = blockDim.y * blockIdx.y + threadIdx.y;	// row

	if (ix >= R[1] || iy >= R[2]) return;					// avoid segfault

	stim::vec3<float> world_pixel;
	world_pixel[0] = (float)ix * S[1];
	world_pixel[1] = (float)iy * S[2];
	world_pixel[2] = ((float)z - Size / 2) * S[3];

	float distance = FLT_MAX;
	float tmp_distance;
	unsigned idx;

	for (unsigned i = 0; i < num; i++) {
		tmp_distance = (V[i].c - world_pixel).len();
		if (tmp_distance <= distance) {
			distance = tmp_distance;
			idx = i;
		}
	}
	if (distance <= V[idx].r) 
		ptr[(R[2] - 1 - iy) * R[0] * R[1] + ix * R[0]] = 255;
}

// indicator for cylinder(envelope/hyperboloid)
__global__ void find_near_cylinder(cylinder* E, unsigned num, size_t* R, float* S, unsigned char* ptr, unsigned z, int Size) {
	
	unsigned ix = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned iy = blockDim.y * blockIdx.y + threadIdx.y;

	if (ix >= R[1] || iy >= R[2]) return;					// avoid segfault

	stim::vec3<float> world_pixel;
	world_pixel[0] = (float)ix * S[1];
	world_pixel[1] = (float)iy * S[2];
	world_pixel[2] = ((float)z - Size / 2) * S[3];

	float distance = FLT_MAX;
	float tmp_distance;
	float rr;												// radii at the surface where projection meets
	
	for (unsigned i = 0; i < num; i++) {					// find the nearest cylinder
		tmp_distance = ((world_pixel - E[i].c1).cross(world_pixel - E[i].c2)).len() / (E[i].c2 - E[i].c1).len();
		if (tmp_distance <= distance) {
			// we only focus on point to line segment
			// check to see whether projection is lying outside the line segment
			float a = (world_pixel - E[i].c1).dot((E[i].c2 - E[i].c1).norm());
			float b = (world_pixel - E[i].c2).dot((E[i].c1 - E[i].c2).norm());
			float length = (E[i].c1 - E[i].c2).len();
			if (a <= length && b <= length) {				// projection lying inside the line segment
				distance = tmp_distance;
				rr = E[i].r1 + (E[i].r2 - E[i].r1) * a / (length);					// linear change
			}
		}
	}
	if (distance <= rr)
		ptr[(R[2] - 1 - iy) * R[0] * R[1] + ix * R[0]] = 255;
}

// make image stack using gpu
void make_image_stack() {

	std::cout << "[-----ON PROGRESS-----]" << std::endl;
	// initilize the image stack object
	I.init(1, size_x, size_y, size_z);
	I.set_dim(dx, dy, dz);

	// because of lack of memory, we have to computer one slice of stack per time
	// allocate vertex and edge
	sphere* d_V;
	cylinder* d_E;

	HANDLE_ERROR(cudaMalloc((void**)&d_V, A.size() * sizeof(sphere)));
	HANDLE_ERROR(cudaMalloc((void**)&d_E, B.size() * sizeof(cylinder)));
	HANDLE_ERROR(cudaMemcpy(d_V, &A[0], A.size() * sizeof(sphere), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(d_E, &B[0], B.size() * sizeof(cylinder), cudaMemcpyHostToDevice));

	// allocate image stack information memory
	float* d_S;
	size_t* d_R;

	size_t* R = (size_t*)malloc(4 * sizeof(size_t));	// size in 4 dimension
	R[0] = 1;
	R[1] = (size_t)size_x;
	R[2] = (size_t)size_y;
	R[3] = (size_t)size_z;
	float* S = (float*)malloc(4 * sizeof(float));		// spacing in 4 dimension
	S[0] = 1.0f;
	S[1] = dx;
	S[2] = dy;
	S[3] = dz;
	size_t num = size_x * size_y;

	HANDLE_ERROR(cudaMalloc((void**)&d_S, 4 * sizeof(float)));
	HANDLE_ERROR(cudaMalloc((void**)&d_R, 4 * sizeof(size_t)));
	HANDLE_ERROR(cudaMemcpy(d_R, R, 4 * sizeof(size_t), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(d_S, S, 4 * sizeof(float), cudaMemcpyHostToDevice));

	// for every slice of image
	unsigned p = 0;																// percentage of progress
	for (unsigned i = 0; i < size_z; i++) {

		// allocate image slice memory
		unsigned char* d_ptr;
		unsigned char* ptr = (unsigned char*)malloc(num * sizeof(unsigned char));
		memset(ptr, 0, num * sizeof(unsigned char));
		
		HANDLE_ERROR(cudaMalloc((void**)&d_ptr, num * sizeof(unsigned char)));
	
		cudaDeviceProp prop;													
		cudaGetDeviceProperties(&prop, 0);										// get cuda device properties structure
		size_t max_thread = sqrt(prop.maxThreadsPerBlock);						// get the maximum number of thread per block

		dim3 block(size_x / max_thread + 1, size_y / max_thread + 1);
		dim3 thread(max_thread, max_thread);
		find_near_sphere << <block, thread >> > (d_V, A.size(), d_R, d_S, d_ptr, i, size_z);
		cudaDeviceSynchronize();
		find_near_cylinder << <block, thread >> > (d_E, B.size(), d_R, d_S, d_ptr, i, size_z);

		HANDLE_ERROR(cudaMemcpy(ptr, d_ptr, num * sizeof(unsigned char), cudaMemcpyDeviceToHost));

		I.set(ptr, i);
	
		free(ptr);
		HANDLE_ERROR(cudaFree(d_ptr));

		// print progress bar
		p = (float)(i + 1) / (float)size_z * 100;
		rtsProgressBar(p);
	}

	// clear up
	free(R);
	free(S);
	HANDLE_ERROR(cudaFree(d_R));
	HANDLE_ERROR(cudaFree(d_S));
	HANDLE_ERROR(cudaFree(d_V));
	HANDLE_ERROR(cudaFree(d_E));
	
	if (stackdir == "")
		I.save_images("image????.bmp");
	else
		I.save_images(stackdir + "/image????.bmp");
	std::cout << std::endl << "[-----SUCCEEDED-----]" << std::endl;
}

// preparation for making image stack
void preparation() {
	
	// clear result from last time
	A.clear();
	B.clear();

	// firstly push back the network
	sphere new_sphere;
	cylinder new_cylinder;

	// push back current network
	for (unsigned i = 0; i < num_vertex; i++) {
		new_sphere.c = V[i].c;
		new_sphere.r = V[i].r;
		A.push_back(new_sphere);
		if (V[i].r > cur_max_radii)
			cur_max_radii = V[i].r;
	}
	for (unsigned i = 0; i < num_edge; i++) {
		new_cylinder.c1 = V[E[i].p[0]].c;
		new_cylinder.c2 = V[E[i].p[1]].c;
		new_cylinder.r1 = V[E[i].p[0]].r;
		new_cylinder.r2 = V[E[i].p[1]].r;
		B.push_back(new_cylinder);
	}

	bridge_start_index = B.size();
	feeder_start_index = A.size();

	// push back the inlet main feeder
	if (inlet_done) {
		new_sphere.c = inlet_port;
		new_sphere.r = main_feeder_radii;
		A.push_back(new_sphere);
		if (main_feeder_radii > cur_max_radii)
			cur_max_radii = main_feeder_radii;
	}

	// push back the outlet main feeder
	if (outlet_done) {
		new_sphere.c = outlet_port;
		new_sphere.r = main_feeder_radii;
		A.push_back(new_sphere);
	}

	// connect input port to inlet main feeder
	float mid_r;
	float p1;
	float p2;
	stim::vec3<float> center1;
	stim::vec3<float> center2;
	float r1;
	float r2;

	for (unsigned i = 0; i < inlet.size(); i++) {
		if (inlet[i].V.size() == 2) {							// straight connection
			mid_r = 2 * inlet[i].r - 1.0f / 2.0f * (V[inlet[i].v[0]].r + default_radii);		// mid_r = 2*ave_r - 1/2(r1 + r2), set proportion to be half
			if (mid_r > cur_max_radii)
				cur_max_radii = mid_r;

			// calculate the envelope along the inlet
			// first half
			center1 = (inlet[i].V[0] + inlet[i].V[1]) / 2;		// normally, the radii of middle point is the largest among those two
			center2 = inlet[i].V[0];
			r1 = mid_r;
			r2 = default_radii;

			// push back middle point
			new_sphere.c = center1;
			new_sphere.r = mid_r;
			A.push_back(new_sphere);

			// push back current cylinder
			new_cylinder.c1 = center1;
			new_cylinder.c2 = center2;
			new_cylinder.r1 = r1;
			new_cylinder.r2 = r2;
			B.push_back(new_cylinder);

			//second half
			center2 = inlet[i].V[1];
			r2 = V[inlet[i].v[0]].r;

			// push back current cylinder
			new_cylinder.c1 = center1;
			new_cylinder.c2 = center2; 
			new_cylinder.r1 = r1;
			new_cylinder.r2 = r2;
			B.push_back(new_cylinder);
		}
		else {											// broken line connection
			p1 = (inlet[i].V[0] - inlet[i].V[1]).len() / inlet[i].l;	// calculate the two line segments length proportion
			p2 = (inlet[i].V[1] - inlet[i].V[2]).len() / inlet[i].l;
			mid_r = (inlet[i].r - (p1 / 2 * default_radii + p2 / 2 * V[inlet[i].v[0]].r)) * 2;
			if (mid_r > cur_max_radii)
				cur_max_radii = mid_r;

			// first half
			center1 = inlet[i].V[1];
			center2 = inlet[i].V[0];
			r1 = mid_r;
			r2 = default_radii;

			// push back corner point
			new_sphere.c = center1;
			new_sphere.r = mid_r;
			A.push_back(new_sphere);

			// push back current cylinder
			new_cylinder.c1 = center1;
			new_cylinder.c2 = center2;
			new_cylinder.r1 = r1;
			new_cylinder.r2 = r2;
			B.push_back(new_cylinder);

			// second half
			center2 = inlet[i].V[2];
			r2 = V[inlet[i].v[0]].r;

			// push back current cylinder
			new_cylinder.c1 = center1;
			new_cylinder.c2 = center2;
			new_cylinder.r1 = r1;
			new_cylinder.r2 = r2;
			B.push_back(new_cylinder);
		}
	}

	// connect output port to outlet main feeder
	for (unsigned i = 0; i < outlet.size(); i++) {
		if (outlet[i].V.size() == 2) {								// straight connection
			mid_r = 2 * outlet[i].r - 1.0f / 2.0f * (V[outlet[i].v[0]].r + default_radii);		// mid_r = 2*ave_r - 1/2(r1 + r2), set proportion to be half
			if (mid_r > cur_max_radii)
				cur_max_radii = mid_r;

			// calculate the envelope along the inlet
			// first half
			center1 = (outlet[i].V[0] + outlet[i].V[1]) / 2;		// normally, the radii of middle poipnt is the largest of these two
			center2 = outlet[i].V[0];
			r1 = mid_r;
			r2 = default_radii;

			// push back middle point
			new_sphere.c = center1;
			new_sphere.r = mid_r;
			A.push_back(new_sphere);

			// push back current cylinder
			new_cylinder.c1 = center1;
			new_cylinder.c2 = center2;
			new_cylinder.r1 = r1;
			new_cylinder.r2 = r2;
			B.push_back(new_cylinder);

			//second half
			center2 = outlet[i].V[1];
			r2 = V[outlet[i].v[0]].r;

			// push back current cylinder
			new_cylinder.c1 = center1;
			new_cylinder.c2 = center2;
			new_cylinder.r1 = r1;
			new_cylinder.r2 = r2;
			B.push_back(new_cylinder);
		}
		else {											// broken line connection
			p1 = (outlet[i].V[0] - outlet[i].V[1]).len() / outlet[i].l;	// calculate the two line segments length proportion
			p2 = (outlet[i].V[1] - outlet[i].V[2]).len() / outlet[i].l;
			mid_r = (outlet[i].r - (p1 / 2 * default_radii + p2 / 2 * V[outlet[i].v[0]].r)) * 2;
			if (mid_r > cur_max_radii)
				cur_max_radii = mid_r;

			// first half
			center1 = outlet[i].V[1];
			center2 = outlet[i].V[0];
			r1 = mid_r;
			r2 = default_radii;

			// push back corner point
			new_sphere.c = center1;
			new_sphere.r = mid_r;
			A.push_back(new_sphere);

			// push back current cylinder
			new_cylinder.c1 = center1;
			new_cylinder.c2 = center2;
			new_cylinder.r1 = r1;
			new_cylinder.r2 = r2;
			B.push_back(new_cylinder);

			// second half
			center2 = outlet[i].V[2];
			r2 = V[outlet[i].v[0]].r;

			// push back current cylinder
			new_cylinder.c1 = center1;
			new_cylinder.c2 = center2;
			new_cylinder.r1 = r1;
			new_cylinder.r2 = r2;
			B.push_back(new_cylinder);
		}
	}

	// get the size of image stack in pixel
	size_x = X / dx + 1;
	size_y = Y / dy + 1;
	size_z = 2.0f * cur_max_radii / dz;
	size_z += 5;								// expand a little bit
}


//*****************************glut functions*********************************
// dynamically set menu
// @param num: number of current menu options
// @param range: range of option to be set from menu_option list
void glut_set_menu(int num, int range) {

	// remove last time menu options
	for (int i = 1; i < num + 1; i++)
		glutRemoveMenuItem(1);

	// set new menu options
	std::string menu_name;
	for (int i = 1; i < range + 1; i++) {
		menu_name = menu_option[i - 1];
		glutAddMenuEntry(menu_name.c_str(), i);
	}
}

// glut projection setting, do squash transformation(from pyramid to cube)
void glut_projection() {

	glMatrixMode(GL_PROJECTION);					
	glPushMatrix();
	glLoadIdentity();
	X = glutGet(GLUT_WINDOW_WIDTH);
	Y = glutGet(GLUT_WINDOW_HEIGHT);
	glViewport(0, 0, X, Y);							
	float aspect = (float)X / (float)Y;				
	gluPerspective(60, aspect, 0.1, 1000000);		
	glPopMatrix();
}

// glut modelview setting, translate camera to origin
void glut_modelview() {
	
	glMatrixMode(GL_MODELVIEW);
	glPushMatrix();
	glLoadIdentity();
	glPopMatrix();

	stim::vec3<float> eye = cam.getPosition();
	stim::vec3<float> focus = cam.getLookAt();
	stim::vec3<float> up = cam.getUp();

	gluLookAt(eye[0], eye[1], eye[2], focus[0], focus[1], focus[2], up[0], up[1], up[2]);
}

// render vertex as point
void glut_draw_point() {
	
	stim::circle<float> tmp_c;
	tmp_c.rotate(stim::vec3<float>(0.0, 0.0, -1.0));					// model circle waiting to be translated and scaled
	for (unsigned i = 0; i < V.size(); i++) {
		if (grow) {
			if (i >= V.size() - new_num)
				break;
		}
		if (!manufacture) {												// in modes except manufacture mode
			if (Flow.P.empty()) 										// if current vertex hasn't been set initial pressure
				glColor3f(0.992f, 0.859f, 0.780f);						// orange point
			else
				if (Flow.P[i] != 0) {
					stim::vec3<float> new_color;
					new_color[0] = (Flow.P[i] / max_pressure) > 0.5f ? 1.0f : 2.0f * Flow.P[i] / max_pressure;						// red
					new_color[1] = 0.0f;																							// green
					new_color[2] = (Flow.P[i] / max_pressure) > 0.5f ? 1.0f - 2.0f * (Flow.P[i] / max_pressure - 0.5f) : 1.0f;		// blue
					glColor3f(new_color[0], new_color[1], new_color[2]);
				}
				else
					glColor3f(0.5f, 0.5f, 0.5f);						//  gray point

			stim::circle<float> c(V[i].c, V[i].r, stim::vec3<float>(0.0, 0.0, 1.0), tmp_c.U);	// create a circle in order to draw the point
			std::vector<typename stim::vec3<float> > cp = c.glpoints(20);	// get points along the circle
			glBegin(GL_TRIANGLE_FAN);										// draw circle as bunch of triangles
			glVertex2f(V[i].c[0], V[i].c[1]);
			for (unsigned i = 0; i < cp.size(); i++) {
				glVertex2f(cp[i][0], cp[i][1]);
			}
			glEnd();
			glFlush();
		}
	}

	if (!generate_network && !simulation && !manufacture) {
		glColor3f(0.0f, 0.0f, 0.0f);

		if (inlet.size() != 0) {
			// draw the inlet main feeder
			stim::circle<float> c(inlet_port, main_feeder_radii, stim::vec3<float>(0.0, 0.0, 1.0), tmp_c.U);	// create a circle in order to draw the point
			std::vector<typename stim::vec3<float> > cp = c.glpoints(20);	// get points along the circle
			glBegin(GL_TRIANGLE_FAN);
			glVertex2f(inlet_port[0], inlet_port[1]);
			for (unsigned i = 0; i < cp.size(); i++) {
				glVertex2f(cp[i][0], cp[i][1]);
			}
			glEnd();
			glFlush();
		}

		if (outlet.size() != 0) {
			// draw the outlet main feeder
			stim::circle<float> c(outlet_port, main_feeder_radii, stim::vec3<float>(0.0, 0.0, 1.0), tmp_c.U);	// create a circle in order to draw the point
			std::vector<typename stim::vec3<float> > cp = c.glpoints(20);	// get points along the circle
			glBegin(GL_TRIANGLE_FAN);
			glVertex2f(outlet_port[0], outlet_port[1]);
			for (unsigned i = 0; i < cp.size(); i++) {
				glVertex2f(cp[i][0], cp[i][1]);
			}
			glEnd();
			glFlush();
		}
	}
}

// render centerline(edge) as line
void glut_draw_line() {

	stim::vec3<float> ori_v;					// direction vector of current edge
	stim::vec3<float> per_v;					// vector perpendicular to direction vector
	stim::vec3<float> v1;						// four vertices for drawing trapezoid
	stim::vec3<float> v2;
	stim::vec3<float> v3;
	stim::vec3<float> v4;
	for (unsigned i = 0; i < E.size(); i++) {	// for every edge
		ori_v = V[E[i].p[1]].c - V[E[i].p[0]].c;
		ori_v = ori_v.norm();
		per_v[0] = -ori_v[1];					// for x dot y = 0, the best solution is x1 = -y2, y1 = x2
		per_v[1] = ori_v[0];
		per_v[2] = ori_v[2];
		v1 = V[E[i].p[0]].c + V[E[i].p[0]].r * per_v;
		v2 = V[E[i].p[0]].c - V[E[i].p[0]].r * per_v;
		v3 = V[E[i].p[1]].c + V[E[i].p[1]].r * per_v;
		v4 = V[E[i].p[1]].c - V[E[i].p[1]].r * per_v;
		
		if (!manufacture) {
			if (color_bound)					// get corresponding color from color map
				glColor3f((float)color[i * 3 + 0] / 255, (float)color[i * 3 + 1] / 255, (float)color[i * 3 + 2] / 255);
			
			glBegin(GL_QUAD_STRIP);
			if (!color_bound) {
				glEnable(GL_BLEND);									// enable color blend
				glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);	// set blend function
				glColor4f(JACKCP[color[i] * 3 + 0], JACKCP[color[i] * 3 + 1], JACKCP[color[i] * 3 + 2], 0.7f);
			}
			glVertex2f(v1[0], v1[1]);
			glVertex2f(v2[0], v2[1]);
			glVertex2f(v3[0], v3[1]);
			glVertex2f(v4[0], v4[1]);
			glEnd();
			if (!color_bound)
				glDisable(GL_BLEND);
		}
		glFlush();
	}

	if (!generate_network && !simulation && !manufacture) {
		glLineWidth(1);

		if (inlet.size() != 0) {
			for (unsigned i = 0; i < inlet.size(); i++) {
				if (inlet_feasibility[i])
					glColor3f(0.0f, 0.0f, 0.0f);					// white means feasible
				else
					glColor3f(1.0f, 0.0f, 0.0f);					// red means nonfeasible
				glBegin(GL_LINE_STRIP);
				for (unsigned j = 0; j < inlet[i].V.size(); j++) {
					glVertex2f(inlet[i].V[j][0], inlet[i].V[j][1]);
				}
				glEnd();
			}
		}
		if (outlet.size() != 0) {
			for (unsigned i = 0; i < outlet.size(); i++) {
				if (outlet_feasibility[i])
					glColor3f(0.0f, 0.0f, 0.0f);					// white means feasible
				else
					glColor3f(1.0f, 0.0f, 0.0f);					// red means nonfeasible
				glBegin(GL_LINE_STRIP);
				for (unsigned j = 0; j < outlet[i].V.size(); j++) {
					glVertex2f(outlet[i].V[j][0], outlet[i].V[j][1]);
				}
				glEnd();
			}
		}
		glFlush();
	}
}

// render flow rane as triangle
void glut_draw_triangle(float threshold = 0.01f) {

	stim::vec3<float> ori_v;		// edge direction vector
	stim::vec3<float> per_v;		// perpendicular vector of ori_v
	stim::vec3<float> mid_p;		// middle point of current edge
	stim::vec3<float> left;			// left point
	stim::vec3<float> right;		// right point
	stim::vec3<float> top;			// top point

	for (unsigned i = 0; i < E.size(); i++) {
		// find the perpendicular vector of current edge
		ori_v = V[E[i].p[1]].c - V[E[i].p[0]].c;
		ori_v = ori_v.norm();
		per_v[0] = -ori_v[1];
		per_v[1] = ori_v[0];
		per_v[2] = ori_v[2];

		mid_p = (V[E[i].p[0]].c + V[E[i].p[1]].c) / 2;
		left = mid_p + per_v * default_radii / 2;
		right = mid_p - per_v * default_radii / 2;

		if (E[i].v > threshold)
			top = mid_p + ori_v * default_radii * sqrt(3.0f);
		else if(E[i].v < -threshold)
			top = mid_p - ori_v * default_radii * sqrt(3.0f);
		
		if (E[i].v > threshold || E[i].v < -threshold) {
			glColor3f(0.600f, 0.847f, 0.788f);	// lime color
			glBegin(GL_TRIANGLES);
			glVertex2f(left[0], left[1]);
			glVertex2f(right[0], right[1]);
			glVertex2f(top[0], top[1]);
			glEnd();
			glFlush();
		}
	}
}

// render inlet/outlet bridge as cylinder
void glut_draw_bridge() {
	
	glColor3f(0.0f, 0.0f, 0.0f);
	std::vector<typename stim::vec3<float> > cp1(subdivision + 1);
	std::vector<typename stim::vec3<float> > cp2(subdivision + 1);

	// draw spheres on the end/middle of bridge
	for (unsigned i = feeder_start_index; i < A.size(); i++) {
		glPushMatrix();
		glTranslatef(A[i].c[0], A[i].c[1], A[i].c[2]);
		glutSolidSphere(A[i].r, subdivision, subdivision);
		glPopMatrix();
	}

	// draw inlet/outlet bridge
	for (unsigned i = bridge_start_index; i < B.size(); i++) {
		// calculate the envelope caps

		find_envelope(cp1, cp2, B[i].c1, B[i].c2, B[i].r1, B[i].r2);
		glBegin(GL_QUAD_STRIP);
		for (unsigned j = 0; j < cp1.size(); j++) {
			glVertex3f(cp1[j][0], cp1[j][1], cp1[j][2]);
			glVertex3f(cp2[j][0], cp2[j][1], cp2[j][2]);
		}
		glEnd();
	}
	glFlush();
}

// render point as sphere
void glut_draw_sphere() {
	
	glColor3f(0.0f, 0.0f, 0.0f);
	for (unsigned i = 0; i < V.size(); i++) {
		glPushMatrix();
		glTranslatef(V[i].c[0], V[i].c[1], V[i].c[2]);
		glutSolidSphere(V[i].r, subdivision, subdivision);
		glPopMatrix();
	}
	if (inlet.size() != 0) {
		// draw the inlet main feeder
		glPushMatrix();
		glTranslatef(inlet_port[0], inlet_port[1], inlet_port[2]);
		glutSolidSphere(main_feeder_radii, subdivision, subdivision);
		glPopMatrix();
	}

	if (outlet.size() != 0) {
		// draw the outlet main feeder
		glPushMatrix();
		glTranslatef(outlet_port[0], outlet_port[1], outlet_port[2]);
		glutSolidSphere(main_feeder_radii, subdivision, subdivision);
		glPopMatrix();
	}
	glFlush();
}

// render line as cylinder
void glut_draw_cylinder() {

	glColor3f(0.0f, 0.0f, 0.0f);
	stim::vec3<float> tmp_d;
	stim::vec3<float> tmp_n;
	stim::vec3<float> center1;
	stim::vec3<float> center2;
	float r1;
	float r2;
	std::vector<typename stim::vec3<float> > cp1(subdivision + 1);
	std::vector<typename stim::vec3<float> > cp2(subdivision + 1);
	for (unsigned i = 0; i < E.size(); i++) {
		center1 = V[E[i].p[0]].c;
		center2 = V[E[i].p[1]].c;
		r1 = V[E[i].p[0]].r;
		r2 = V[E[i].p[1]].r;

		// calculate the envelope caps
		find_envelope(cp1, cp2, center1, center2, r1, r2);

		glBegin(GL_QUAD_STRIP);
		for (unsigned j = 0; j < cp1.size(); j++) {
			glVertex3f(cp1[j][0], cp1[j][1], cp1[j][2]);
			glVertex3f(cp2[j][0], cp2[j][1], cp2[j][2]);
		}
		glEnd();
		glFlush();
	}
}

// main render function
void glut_render() {

	glEnable(GL_SMOOTH);
	glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glut_draw_line();								// draw the edge as line
	glut_draw_point();								// draw the vertex as point

	if (!first_click && generate_network) {					// render a transparent line to indicate your next click position
		glEnable(GL_BLEND);									// enable color blend
		glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);	// set blend function
		glColor4f(JACKCP[color_index * 3 + 0], JACKCP[color_index * 3 + 1], JACKCP[color_index * 3 + 2], 0.2f);
		stim::vec3<float> tmp_d;
		stim::circle<float> tmp_c;
		std::vector<typename stim::vec3<float> > cp1(subdivision + 1);
		std::vector<typename stim::vec3<float> > cp2(subdivision + 1);
		tmp_d = tmp_vertex.c - V[tmp_edge.p[0]].c;
		tmp_d = tmp_d.norm();
		tmp_c.rotate(tmp_d);
		stim::circle<float> c1(V[tmp_edge.p[0]].c, V[tmp_edge.p[0]].r, tmp_d, tmp_c.U);
		stim::circle<float> c2(tmp_vertex.c, tmp_vertex.r, tmp_d, tmp_c.U);
		cp1 = c1.glpoints(subdivision);
		cp2 = c2.glpoints(subdivision);
		glBegin(GL_QUAD_STRIP);
		for (unsigned j = 0; j < subdivision + 1; j++) {
			glVertex3f(cp1[j][0], cp1[j][1], cp1[j][2]);
			glVertex3f(cp2[j][0], cp2[j][1], cp2[j][2]);
		}
		glEnd();
		glFlush();
		glDisable(GL_BLEND);
	}
	
	if (grow) {										// render a gray line to indicate grow edge
		glColor3f(0.5f, 0.5f, 0.5f);
		glBegin(GL_LINES);
		glVertex2f(V[tmp_edge.p[0]].c[0], V[tmp_edge.p[0]].c[1]);
		glVertex2f(tmp_vertex.c[0], tmp_vertex.c[1]);
		glEnd();

		// render the new edges and new vertex
		for (unsigned i = num_vertex; i < V.size(); i++) {
			glPointSize(10);
			glBegin(GL_POINT);
			glVertex2f(V[i].c[0], V[i].c[1]);
			glEnd();
		}
		for (unsigned i = 0; i < tmp_E.size(); i++) {
			glBegin(GL_LINES);
			glVertex2f(V[tmp_E[i].p[0]].c[0], V[tmp_E[i].p[0]].c[1]);
			glVertex2f(V[tmp_E[i].p[1]].c[0], V[tmp_E[i].p[1]].c[1]);
			glEnd();
		}
		glFlush();
	}

	if (select_corner) {
		glEnable(GL_BLEND);									// enable color blend
		glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);	// set blend function
		glColor4f(0.0f, 0.0f, 0.0f, 0.4f);
		
		// draw the joint position as a point
		glBegin(GL_POINT);
		glVertex2f(corner_vertex[0], corner_vertex[1]);
		glEnd();

		// draw the bridge
		glBegin(GL_LINE_STRIP);
		if (build_inlet) {
			glVertex2f(inlet[bridge_index].V[0][0], inlet[bridge_index].V[0][1]);
			glVertex2f(corner_vertex[0], corner_vertex[1]);
			unsigned idx = inlet[bridge_index].V.size() - 1;
			glVertex2f(inlet[bridge_index].V[idx][0], inlet[bridge_index].V[idx][1]);
		}
		else if (build_outlet) {
			glVertex2f(outlet[bridge_index].V[0][0], outlet[bridge_index].V[0][1]);
			glVertex2f(corner_vertex[0], corner_vertex[1]);
			unsigned idx = outlet[bridge_index].V.size() - 1;
			glVertex2f(outlet[bridge_index].V[idx][0], outlet[bridge_index].V[idx][1]);
		}
		glEnd();
		glFlush();
		glDisable(GL_BLEND);
	}

	if (!manufacture) {

		if (simulation || build_inlet || build_outlet) {
			glut_draw_triangle();
		}

		for (unsigned i = 0; i < V.size(); i++) {
			glColor3f(0.0f, 0.0f, 0.0f);					
			glRasterPos2f(V[i].c[0], V[i].c[1] + 0.5f);	// mark index right above the vertex
			std::stringstream ss;
			ss << i;
			glutBitmapString(GLUT_BITMAP_HELVETICA_18, (const unsigned char*)(ss.str().c_str()));
		}

		// bring up a pressure bar on left
		if (select_pressure) {
			glLineWidth(100);
			glBegin(GL_LINES);
			glColor3f(0.0f, 0.0f, 1.0f);				// blue to red
			glVertex2f(border * X / vX, border * Y / vY);
			glColor3f(1.0, 0.0, 0.0);
			glVertex2f(border * X / vX, (vY - 2 * border) * Y / vY);
			glEnd();
			glFlush();

			// pressure bar text
			glColor3f(0.0f, 0.0f, 0.0f);
			glRasterPos2f(0.0f, (vY - border) * Y / vY);
			std::stringstream ss_p;
			ss_p << "Pressure Bar";
			glutBitmapString(GLUT_BITMAP_HELVETICA_18, (const unsigned char*)(ss_p.str().c_str()));

			// pressure range text
			float step = vY - 3 * border;
			step /= 10;
			for (unsigned i = 0; i < 11; i++) {
				glRasterPos2f((border * 1.5f) * X / vX, (border + i * step) * Y / vY);
				std::stringstream ss_n;
				ss_n << i * max_pressure / 10;
				glutBitmapString(GLUT_BITMAP_HELVETICA_18, (const unsigned char*)(ss_n.str().c_str()));
			}
		}
	}

	// print the velocity range bar
	if (simulation && !select_pressure) {
		if (dangle_vertex.size() == 2 && num_edge - num_vertex + 1 <= 0) {
			// do nothing
		}
		else {
			float step = (vY - 3 * border) * Y / vY;
			step /= BREWER_CTRL_PTS - 1;
			for (unsigned i = 0; i < BREWER_CTRL_PTS - 1; i++) {
				glLineWidth(100);
				glBegin(GL_LINES);
				glColor3f(BREWERCP[i * 4 + 0], BREWERCP[i * 4 + 1], BREWERCP[i * 4 + 2]);
				glVertex2f(border * X / vX, border * Y / vY + i * step);
				glColor3f(BREWERCP[(i + 1) * 4 + 0], BREWERCP[(i + 1) * 4 + 1], BREWERCP[(i + 1) * 4 + 2]);
				glVertex2f(border * X / vX, border * Y / vY + (i + 1) * step);
				glEnd();
			}
			glFlush();

			// pressure bar text
			glColor3f(0.0f, 0.0f, 0.0f);
			glRasterPos2f(0.0f, (vY - border) * Y / vY);
			std::stringstream ss_p;
			ss_p << "Velocity range";
			glutBitmapString(GLUT_BITMAP_HELVETICA_18, (const unsigned char*)(ss_p.str().c_str()));

			// pressure range text
			step = vY - 3 * border;
			step /= 10;
			for (unsigned i = 0; i < 11; i++) {
				glRasterPos2f((border * 1.5f) * X / vX, (border + i * step) * Y / vY);
				std::stringstream ss_n;
				ss_n << min_v + i * (max_v - min_v) / 10;
				glutBitmapString(GLUT_BITMAP_HELVETICA_18, (const unsigned char*)(ss_n.str().c_str()));
			}
		}
	}

	if (manufacture) {
		glut_draw_sphere();
		glut_draw_cylinder();
		glut_draw_bridge();
	}

	if (radii_changed) {
		glColor3f(0.835f, 0.243f, 0.310f);		
		glRasterPos2f(V[radii_index].c[0], V[radii_index].c[1] - 1.0f);
		std::stringstream ss_r;
		ss_r << "r=" << V[radii_index].r;
		glutBitmapString(GLUT_BITMAP_HELVETICA_18, (const unsigned char*)(ss_r.str().c_str()));
		radii_changed = false;
	}

	glutSwapBuffers();
}

// register mouse click events
void glut_mouse(int button, int state, int x, int y) {

	if (button == GLUT_RIGHT_BUTTON)
		return;

	mods = glutGetModifiers();					// get special keyboard input

	if (button == GLUT_LEFT_BUTTON && state == GLUT_DOWN) {
		std::cout << "\r";						// clear up ERROR reminder
		std::cout << "\t\t\t\t\t\t\t\t\t";
		std::cout.flush();
	}

	// to generate a new network by mouse click
	if (button == GLUT_LEFT_BUTTON && state == GLUT_DOWN && generate_network && mods == 0) {

		mouse_x = x;							// get the click position in the window coordinates
		mouse_y = y;
		unsigned idx = UINT_MAX;				// stores the vertex's index

		if (first_click) {						// first click of one line of edge
			flag = epsilon_vertex(x, y, idx);	// find out whether current position appears a vertex
			if (flag) {
				new_edge.p[0] = idx;			// store the geometry start vertex index
				tmp_edge.p[0] = idx;
				num++;
			}
			else {
				new_vertex.c = stim::vec3<float>(x, (vY - y), 0);	// make a new vertex
				new_vertex.c[0] = new_vertex.c[0] * (float)X / vX;
				new_vertex.c[1] = new_vertex.c[1] * (float)Y / vY;
				new_edge.p[0] = iter;								// make a new edge and set the starting vertex
				tmp_edge.p[0] = iter;
				V.push_back(new_vertex);							// push a new vertex
				iter++;												// iterator + 1
				num++;												// added a vertex
			}
			first_click = false;									// finished first click
		}
		else {								// following click of one line of edge
			flag = epsilon_vertex(x, y, idx);
			if (flag) {
				if (!is_edge(idx)) {		// no edge between two vertices
					if (idx != UINT_MAX) {	// acceptable click
						new_edge.p[1] = idx;
						if (new_edge.p[0] != new_edge.p[1]) {	// simple graph, no loop and parallel edge
							E.push_back(new_edge);
							color.push_back(color_index);		// record the color scheme
							first_click = true;
							num = 0;							// start a new line of edges
							color_index = (color_index == JACK_CTRL_PTS - 1) ? 0 : color_index + 1;	// update color scheme for new line of edges
						}
						else {
							Sleep(100);							// make flash effect
							std::cout << "\r";
							std::cout << "[  ERROR  ]  ";
							std::cout << "You can't do that!";
							std::cout.flush();
						}
					}
				}
				else {
					Sleep(100);				// make flash effect
					std::cout << "\r";
					std::cout << "[  ERROR  ]  ";
					std::cout << "There exists an edge between these two vertices";
					std::cout.flush();
				}
			}
			else {
				new_vertex.c = stim::vec3<float>(x, (vY - y), 0);	// make a new vertex
				new_vertex.c[0] = new_vertex.c[0] * (float)X / vX;
				new_vertex.c[1] = new_vertex.c[1] * (float)Y / vY;
				new_edge.p[1] = iter;								// make a new edge and set the starting vertex to current
				V.push_back(new_vertex);							// push a new vertex
				E.push_back(new_edge);								// push a new edge
				color.push_back(color_index);						// record the color scheme
				new_edge.p[0] = iter;								// make a new edge and set the starting vertex to current
				tmp_edge.p[0] = iter;
				iter++;												// iterator + 1
				num++;												// added a vertex
			}
		}
	}

	// modify pressure
	if (button == GLUT_LEFT_BUTTON && state == GLUT_DOWN && simulation && mods == 0 && !grow) {

		mouse_x = x;
		mouse_y = y;
		if (select_pressure) {							// if a vertex had been selected to be modified pressure
			if (vY - y < border || vY - y > vY - 2 * border) {	// click outside the bar along y-axis
				Sleep(100);								// make flash effect
				std::cout << "\r";
				std::cout << "[  ERROR  ]  ";
				std::cout << "Click exceeds the range of pressure bar";
				std::cout.flush();
			}
			else {
				select_pressure = false;				// finished setting the pressure of chose vertex
				Flow.P[pressure_index] = (vY - mouse_y - border) / (vY - 3 * border) * max_pressure;	// get the pressure value on pressure bar

				system("CLS");							// clear up console box
				std::cout << " ===================" << std::endl;
				std::cout << "|  SIMULATION MODE  |" << std::endl;
				std::cout << " ===================" << std::endl << std::endl;
				std::cout << "[  TIP  ]  ";
				std::cout << "Click dangle vertex to set pressure" << std::endl;
				std::cout << "           Move wheel to change radii of the vertex which the cursor meets" << std::endl;

				// simulate again
				find_stable_state();
				show_stable_state();
			}
		}
		else {
			unsigned tmp_p = 0;
			bool flag = epsilon_vertex(mouse_x, mouse_y, tmp_p);
			if (flag) {
				std::vector<unsigned>::iterator it = std::find(dangle_vertex.begin(), dangle_vertex.end(), tmp_p);
				if (it == dangle_vertex.end()) {		// if it is not dangle vertex
					Sleep(100);							// make flash effect
					std::cout << "\r";
					std::cout << "[  ERROR  ]  ";
					std::cout << "Only dangle vertex pressure need to be set";
					std::cout.flush();
				}
				else {									// if it is dangle vertex
					select_pressure = true;				// set flag to true
					pressure_index = tmp_p;				// stores the index of vertex
				}
			}
		}
	}

	// build inlet and outlet
	if (button == GLUT_LEFT_BUTTON && state == GLUT_DOWN && (build_inlet || build_outlet) && !select_bridge && !select_corner && mods == 0) {

		mouse_x = x;
		mouse_y = y;

		select_bridge = true;

		if (build_inlet) {
			inlet_port = stim::vec3<float>(x, (vY - y), 0);			// get the inlet port coordinates
			inlet_port[0] = inlet_port[0] * (float)X / vX;
			inlet_port[1] = inlet_port[1] * (float)Y / vY;
			inlet_done = true;

			float tmp_l;
			for (unsigned i = 0; i < input.size(); i++) {
				stim::bridge<float> b;
				// push back vertices
				b.V.push_back(inlet_port);
				b.V.push_back(V[input[i].first].c);

				// one direct line
				tmp_l = (inlet_port - V[input[i].first].c).len();

				b.Q = input[i].third;
				b.l = tmp_l;
				b.v.push_back(input[i].first);						// only store the dangle vertex index information
				inlet.push_back(b);
			}

			// check out current connection
			is_acceptable();
		}
		else if (build_outlet) {
			outlet_port = stim::vec3<float>(x, (vY - y), 0);		// get the inlet port coordinates
			outlet_port[0] = outlet_port[0] * (float)X / vX;
			outlet_port[1] = outlet_port[1] * (float)Y / vY;
			outlet_done = true;

			float tmp_l;
			for (unsigned i = 0; i < output.size(); i++) {
				stim::bridge<float> b;
				// push back vertices
				b.V.push_back(outlet_port);
				b.V.push_back(V[output[i].first].c);

				// one direct line
				tmp_l = (outlet_port - V[output[i].first].c).len();

				b.Q = output[i].third;
				b.l = tmp_l;
				b.v.push_back(output[i].first);
				outlet.push_back(b);
			}

			// check out current connection
			is_acceptable();
		}
	}

	// select a bridge to modify
	else if (button == GLUT_LEFT_BUTTON && state == GLUT_DOWN && select_bridge && mods == 0) {

		mouse_x = x;
		mouse_y = y;

		bool flag = epsilon_edge(mouse_x, mouse_y, bridge_index);
		if (flag) {
			select_bridge = false;
			select_corner = true;
		}
		else {
			Sleep(100);						// make flash effect
			std::cout << "\r";
			std::cout << "[  ERROR  ]  ";
			std::cout << "No bridge at where your click";
		}
	}

	// re connect the inlet/outlet that selected
	else if (button == GLUT_LEFT_BUTTON && state == GLUT_DOWN && select_corner && mods == 0) {

		mouse_x = x;
		mouse_y = y;

		mask_done = false;										// recalculate the connection

		corner_vertex = stim::vec3<float>(x, (vY - y), 0);		// get the corner vertex
		corner_vertex[0] = corner_vertex[0] * (float)X / vX;
		corner_vertex[1] = corner_vertex[1] * (float)Y / vY;

		if (build_inlet) {
			stim::bridge<float> tmp_b;
			tmp_b.V.push_back(inlet_port);						// push back the inlet port vertex
			tmp_b.V.push_back(corner_vertex);					// push back the corner vertex
			unsigned idx = inlet[bridge_index].V.size() - 1;	// get the dangle vertex index from the inlet
			tmp_b.V.push_back(inlet[bridge_index].V[idx]);		// push back the dangle vertex
			tmp_b.l = (tmp_b.V[0] - tmp_b.V[1]).len() + (tmp_b.V[1] - tmp_b.V[2]).len();
			tmp_b.Q = inlet[bridge_index].Q;
			tmp_b.v.push_back(inlet[bridge_index].v[0]);
			tmp_b.r = inlet[bridge_index].r;

			inlet[bridge_index] = tmp_b;
		}

		else if (build_outlet) {
			stim::bridge<float> tmp_b;
			tmp_b.V.push_back(outlet_port);						// push back the inlet port vertex
			tmp_b.V.push_back(corner_vertex);					// push back the corner vertex
			unsigned idx = outlet[bridge_index].V.size() - 1;	// get the dangle vertex index from the outlet
			tmp_b.V.push_back(outlet[bridge_index].V[idx]);		// push back the dangle vertex
			tmp_b.l = (tmp_b.V[0] - tmp_b.V[1]).len() + (tmp_b.V[1] - tmp_b.V[2]).len();
			tmp_b.Q = outlet[bridge_index].Q;
			tmp_b.v.push_back(outlet[bridge_index].v[0]);
			tmp_b.r = outlet[bridge_index].r;

			outlet[bridge_index] = tmp_b;
		}

		// check out current connection
		is_acceptable();

		select_corner = false;
		select_bridge = true;
	}

	// left CTRL + left mouse to grow a line new edges from any vertex
	if (button == GLUT_LEFT_BUTTON && state == GLUT_DOWN && simulation && mods == GLUT_ACTIVE_CTRL && grow) {

		mouse_x = x;
		mouse_y = y;

		unsigned i;

		bool flag = epsilon_edge(mouse_x, mouse_y, edge_index, i);
		if (flag) {
			for (unsigned j = 0; j < tmp_E.size(); j++)
				E.push_back(tmp_E[j]);
			new_vertex = V[E[edge_index].p[i]];
			new_edge.p[1] = E[edge_index].p[i];
			E.push_back(new_edge);

			get_background();		// get network basic information
			flow_initialize();		// initialize flow

			find_stable_state();	// main function of solving the linear system
			show_stable_state();	// output results as csv files
			grow = false;
		}
		else {
			new_vertex.c = stim::vec3<float>(x, (vY - y), 0);	// make a new vertex
			new_vertex.c[0] = new_vertex.c[0] * (float)X / vX;
			new_vertex.c[1] = new_vertex.c[1] * (float)Y / vY;
			unsigned num = V.size();							// get the new vertex index
			V.push_back(new_vertex);
			new_edge.p[1] = num;
			tmp_E.push_back(new_edge);
			new_edge.p[0] = num;
			tmp_edge.p[0] = num;
			new_num++;
		}
	}
	else if (button == GLUT_LEFT_BUTTON && state == GLUT_DOWN && simulation && mods == GLUT_ACTIVE_CTRL && !grow) {
		
		mouse_x = x;
		mouse_y = y;

		// new point information
		unsigned i;
		new_num = 0;

		bool flag = epsilon_edge(mouse_x, mouse_y, edge_index, i);
		if (flag) {
			grow = true;
			new_vertex = V[E[edge_index].p[i]];
			new_edge.p[0] = E[edge_index].p[i];
			tmp_edge.p[0] = E[edge_index].p[i];
		}
		else {
			Sleep(100);						// make flash effect
			std::cout << "\r";
			std::cout << "[  ERROR  ]  ";
			std::cout << "No vertex at where your click";
		}
	}
}

// register mouse move events
void glut_motion(int x, int y) {

	tmp_vertex.c = stim::vec3<float>(x, (vY - y), 0);
	tmp_vertex.c[0] = tmp_vertex.c[0] * (float)X / vX;
	tmp_vertex.c[1] = tmp_vertex.c[1] * (float)Y / vY;

	corner_vertex[0] = tmp_vertex.c[0];
	corner_vertex[1] = tmp_vertex.c[1];

	glutPostRedisplay();
}

// register wheel events
void glut_wheel(int wheel, int direction, int x, int y) {

	std::cout << "\r";							// clear up ERROR reminder
	std::cout << "\t\t\t\t\t\t\t\t\t";
	std::cout.flush();

	if (simulation) {
		flag = epsilon_vertex(x, y, radii_index);
		if (flag) {
			radii_changed = true;
			if (direction > 0)					// increase radii
				V[radii_index].r += radii_factor;
			else {
				V[radii_index].r -= radii_factor;
				if (V[radii_index].r <= 0) {			// degenerate case where radii less than 0
					Sleep(100);					// make flash effect
					std::cout << "\r";
					std::cout << "[  ERROR  ]  ";
					std::cout << "Radii is less than 0, reset to default radii";
					V[radii_index].r = default_radii;
				}	
			}
		}
	
		system("CLS");							// clear up console box
		std::cout << " ===================" << std::endl;
		std::cout << "|  SIMULATION MODE  |" << std::endl;
		std::cout << " ===================" << std::endl << std::endl;
		std::cout << "[  TIP  ]  ";
		std::cout << "Click dangle vertex to set pressure" << std::endl;
		std::cout << "           Move wheel to change radii of the vertex which the cursor meets" << std::endl;

		// simulate again
		find_stable_state();
		show_stable_state();
	}

	glutPostRedisplay();
}

// register keyboard inputs
void glut_keyboard(unsigned char key, int x, int y) {
	
	switch (key) {
	// press space to start a new line of edges
	case 32:
		first_click = true;
		num = 0;
		color_index = (color_index == JACK_CTRL_PTS - 1) ? 0 : color_index + 1;	// update color scheme for new line of edges
		break;

	// reset main feeder position
	case 'c':
		if (build_inlet || build_outlet) {
			select_bridge = false;
			select_corner = false;
			if (build_inlet) {
				inlet_done = false;
				inlet.clear();
			}
			else if (build_outlet) {
				outlet_done = false;
				outlet.clear();
			}
			mask_done = false;
		}
		break;

	// output the image stack
	case 'm':
		if (manufacture) {
#ifdef __CUDACC__
			make_image_stack();
#else
			std::cout << "You need to have a gpu to make image stack, sorry." << std::endl;
#endif
		}
		break;

	// output the drawn network
	case 's': 
	{
		stringstream output_ss;
		output_ss << name << "_" << sub_name << "_net" << ".obj";
		std::string output_filename = output_ss.str();
		std::ofstream output_file;

		output_file.open(output_filename.c_str());
		for (unsigned i = 0; i < V.size(); i++)
			output_file << "v" << " " << V[i].c[0] << " " << V[i].c[1] << " " << V[i].c[2] << std::endl;
		for (unsigned i = 0; i < V.size(); i++)
			output_file << "vt" << " " << V[i].r << std::endl;
		for (unsigned i = 0; i < E.size(); i++)
			output_file << "l" << " " << E[i].p[0] + 1 << "/" << E[i].p[0] + 1 << " " << E[i].p[1] + 1 << "/" << E[i].p[1] + 1 << std::endl;
		output_file.close();
		sub_name++;			// sub name change
		break;
	}

	// undo
	case 'u': {

		// first vertex on a new line of edges
		if (num == 1) {
			bool flag = false;						// check whether current vertex belongs to another edge
			for (unsigned i = 0; i < E.size(); i++) {
				if (new_edge.p[0] == E[i].p[0] || new_edge.p[0] == E[i].p[1]) {
					flag = true;
					break;
				}
			}
			if (new_edge.p[0] == V.size() - 1 && !flag) {	// new vertex
				V.pop_back();								// pop back new vertex
				iter--;
			}
			first_click = true;
			num = 0;
		}
		// not first vertex
		else if (num > 1) {
			new_edge.p[0] = E[E.size() - 1].p[0];
			tmp_edge.p[0] = new_edge.p[0];
			E.pop_back();							// pop back new "things"
			color.pop_back();
			V.pop_back();
			iter--;
			num--;
		}
		break;
	}

	// close window and exit application
	case 27:						// if keyboard 'ESC' is pressed, then exit
		std::exit(1);
	}

	glutPostRedisplay();
}

// register glut menu options
void glut_menu(int value) {

	cur_menu_num = glutGet(GLUT_MENU_NUM_ITEMS);

	if (value == 1) {				// generation mode

		system("CLS");				// clear up console
		std::cout << " ==================" << std::endl;
		std::cout << "|  GENERATOR MODE  |" << std::endl;
		std::cout << " ==================" << std::endl << std::endl;
		std::cout << "[  TIP  ]  ";
		std::cout << "Click to draw a network. (press SPACE to start a new line of edges)" << std::endl;
		
		// clear up previous work
		glClear(GL_COLOR_BUFFER_BIT);
		V.clear();
		E.clear();
		iter = 0;
		num = 0;

		// set up flags
		generate_network = true;
		simulation = false;
		manufacture = false;
		first_click = true;
		build_inlet = false;
		build_outlet = false;
		select_bridge = false;
		mask_done = false;
		color_bound = false;
		name++;							// name sequence increments

		new_menu_num = 2;				// set new menu option number
	}

	if (value == 2) {							// simulation mode

		// clear previous drawn buffer
		glClear(GL_COLOR_BUFFER_BIT);
		iter = 0;
		num = 0;

		system("CLS");							// clear up console box
		std::cout << " ===================" << std::endl;
		std::cout << "|  SIMULATION MODE  |" << std::endl;
		std::cout << " ===================" << std::endl << std::endl;
		std::cout << "[  TIP  ]  ";
		std::cout << "Click dangle vertex to set pressure" << std::endl;
		std::cout << "           Move wheel to change radii of the vertex which the cursor meets" << std::endl;

		// set up flags
		generate_network = false;
		simulation = true;
		manufacture = false;
		build_inlet = false;
		build_outlet = false;
		select_bridge = false;
		mask_done = false;

		if (first_simulation) {
			get_background();		// get network basic information
			flow_initialize();		// initialize flow
			first_simulation = false;
		}

		// set other initial information then solve the network
		find_stable_state();					// main function of solving the linear system
		show_stable_state();					// output results as csv files

		// set the camera object
		stim::vec3<float> c = (L + U) * 0.5f;		// get the center of the bounding box
		stim::vec3<float> size = (U - L);			// get the size of the bounding box

		// place the camera along the z-axis at a distance determined by the network size along x and y
		cam.setPosition(c + stim::vec<float>(0, 0, camera_factor * std::max(size[0], size[1])));
		cam.LookAt(c[0], c[1], c[2]);

		new_menu_num = 5;						// set new menu option number
	}

	if (value == 3) {							// building inlet mode
		
		system("CLS");							// clear up console
		std::cout << " ====================" << std::endl;
		std::cout << "|  BUILD INLET MODE  |" << std::endl;
		std::cout << " ====================" << std::endl << std::endl;
		std::cout << "[  TIP  ]  ";
		std::cout << "Firstly, click any position to set inlet main feeder" << std::endl;
		std::cout << "           Then, click any bridge to translocate" << std::endl;
		std::cout << "           System will check and print current bridge status :)" << std::endl;
		std::cout << "           Press c to delete inlet main feeder and bridges" << std::endl;
		std::cout << "           If current bridge is not acceptable, you can either do:" << std::endl;
		std::cout << "            [*1. increase the pressure at the vertex which is pointed out" << std::endl;
		std::cout << "              2. increase the length of connection at that vertex" << std::endl;
		std::cout << "              3. use more advance manufacture machine]" << std::endl;
		std::cout << "[  NOTE  ] ";
		std::cout << "Delete main feeder before modify if you want to change input ports" << std::endl << std::endl;

		// set up flags
		if (!inlet_done) {						// first time need to set main feeder position
			generate_network = false;
			simulation = false;
			manufacture = false;
			build_inlet = true;
			build_outlet = false;
			select_pressure = false;
			select_bridge = false;
			select_corner = false;
			mask_done = false;
		}
		else {									// already set the inlet main feeder position
			generate_network = false;
			simulation = false;
			manufacture = false;
			build_inlet = true;
			build_outlet = false;
			select_pressure = false;
			select_bridge = true;

			// check out current connection
			is_acceptable();
		}
		new_menu_num = 5;						// set new menu option number
	}

	if (value == 4) {							// building outlet mode
		
		system("CLS");							// clear up console box
		std::cout << " =====================" << std::endl;
		std::cout << "|  BUILD OUTLET MODE  |" << std::endl;
		std::cout << " =====================" << std::endl << std::endl;
		std::cout << "[  TIP  ]  ";
		std::cout << "Firstly, click any position to set inlet main feeder" << std::endl;
		std::cout << "           Then, click any bridge to translocate" << std::endl;
		std::cout << "           System will check and print current bridge status :)" << std::endl;
		std::cout << "           Press c to delete outlet main feeder and bridges" << std::endl;
		std::cout << "           If current bridge is not acceptable, you can either do:" << std::endl;
		std::cout << "            [*1. decrease the pressure at the vertex which is pointed out" << std::endl;
		std::cout << "              2. increase the length of connection at that vertex" << std::endl;
		std::cout << "              3. use more advance manufacture machine]" << std::endl;
		std::cout << "[  NOTE  ] ";
		std::cout << "Delete main feeder before modify if you want to change output ports" << std::endl << std::endl;

		// set up flags
		if (!outlet_done) {						// first time need to set main feeder position
			generate_network = false;
			simulation = false;
			manufacture = false;
			build_inlet = false;
			build_outlet = true;
			select_pressure = false;
			select_bridge = false;
			select_corner = false;
			mask_done = false;
		}
		else {									// already set the outlet main feeder position
			generate_network = false;
			simulation = false;
			manufacture = false;
			build_inlet = false;
			build_outlet = true;
			select_bridge = true;
			select_pressure = false;
			select_corner = false;

			// check out current connection
			is_acceptable();
		}
		new_menu_num = 5;						// set new menu option number
	}

	if (value == 5) {							// manufacture mode
		
		system("CLS");							// clear up console box
		std::cout << " ====================" << std::endl;
		std::cout << "|  MANUFACTURE MODE  |" << std::endl;
		std::cout << " ====================" << std::endl << std::endl;
		std::cout << "[  TIP  ]  ";
		std::cout << "Press m to make and save image stack" << std::endl;
		
		// set up flags
		generate_network = false;
		simulation = false;
		manufacture = true;
		build_inlet = false;
		build_outlet = false;
		select_bridge = false;

		if (!mask_done) {
			// calculate the inlet connection radii
			unsigned midx;
			for (unsigned i = 0; i < inlet.size(); i++) {
				if (inlet[i].v[0] == min_input_index) {
					midx = i;
					break;
				}
			}
			for (unsigned i = 0; i < inlet.size(); i++) {
				unsigned idx = inlet[i].v[0];

				if (idx == min_input_index) {
					inlet[i].r = minimum_radii;				// set the maximum pressure connection to minimum radii
				}
				else {										// P1 + deltaP1 = P2 + deltaP2
					float tmp_r;
					if (mode == 2) {
						tmp_r = (Flow.pressure[min_input_index] + ((12 * u * inlet[midx].l * inlet[midx].Q) / (std::pow(h, 3) * 2 * minimum_radii)) - Flow.pressure[idx]) * (std::pow(h, 3)) / (12 * u * inlet[i].l * inlet[i].Q);
						tmp_r = (1 / tmp_r) / 2;
					}
					else if (mode == 3) {
						tmp_r = (Flow.pressure[min_input_index] + ((8 * u * inlet[midx].l * inlet[midx].Q) / (std::pow(minimum_radii, 4) * (float)stim::PI)) - Flow.pressure[idx]) * (float)stim::PI / (8 * u * inlet[i].l * inlet[i].Q);
						tmp_r = std::pow(1 / tmp_r, 1.0f / 4);
					}
					inlet[i].r = tmp_r;
				}
			}

			// calculate the outlet connection radii
			for (unsigned i = 0; i < outlet.size(); i++) {
				if (outlet[i].v[0] == max_output_index) {
					midx = i;
					break;
				}
			}
			for (unsigned i = 0; i < outlet.size(); i++) {
				unsigned idx = outlet[i].v[0];
				if (idx == max_output_index) {
					outlet[i].r = minimum_radii;			// set the maximum pressure connection to minimum radii
				}
				else {										// P1 - deltaP1 = P2 - deltaP2
					float tmp_r;
					if (mode == 2) {
						tmp_r = (Flow.pressure[idx] - (Flow.pressure[max_output_index] - (12 * u * outlet[midx].l * outlet[midx].Q) / (std::pow(h, 3) * 2 * minimum_radii))) * (std::pow(h, 3)) / (12 * u * outlet[i].l * outlet[i].Q);
						tmp_r = (1 / tmp_r) / 2;
					}
					else if (mode == 3) {
						tmp_r = (Flow.pressure[idx] - (Flow.pressure[max_output_index] - (8 * u * outlet[midx].l * outlet[midx].Q) / (std::pow(minimum_radii, 4) * (float)stim::PI))) * (float)stim::PI / (8 * u * outlet[i].l * outlet[i].Q);
						tmp_r = std::pow(1 / tmp_r, 1.0f / 4);
					}
					outlet[i].r = tmp_r;
				}
			}
		}

		inlet_flow_rate = outlet_flow_rate = 0.0f;
		// calculate the main feeder flow rate and pressure
		for (unsigned i = 0; i < inlet.size(); i++) {
			inlet_flow_rate += fabsf(inlet[i].Q);
		}
		for (unsigned i = 0; i < outlet.size(); i++) {
			outlet_flow_rate += fabsf(outlet[i].Q);
		}
		for (unsigned i = 0; i < inlet.size(); i++) {
			unsigned idx = inlet[i].v[0];
			if (mode == 2)
				inlet_pressure = Flow.pressure[idx] + (12 * u * inlet[i].l * inlet[i].Q) / (2 * inlet[i].r * std::pow(h, 3));
			else if (mode == 3)
				inlet_pressure = Flow.pressure[idx] + (8 * u * inlet[i].l * inlet[i].Q) / ((float)stim::PI * std::pow(inlet[i].r, 4));
		}
		for (unsigned i = 0; i < outlet.size(); i++) {
			unsigned idx = outlet[i].v[0];
			if (mode == 2)
				outlet_pressure = Flow.pressure[idx] - (12 * u * outlet[i].l * outlet[i].Q) / (2 * inlet[i].r * std::pow(h, 3));
			else if (mode == 3)
				outlet_pressure = Flow.pressure[idx] - (8 * u * outlet[i].l * outlet[i].Q) / ((float)stim::PI * std::pow(outlet[i].r, 4));
		}

		mask_done = true;
		preparation();										// preparation for making image stack

		new_menu_num = 5;									// set new menu option number
	}

	// set up new menu
	glut_set_menu(cur_menu_num, new_menu_num);

	glutPostRedisplay();
}

// window reshape function
void glut_reshape(int x, int y) {
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	vX = glutGet(GLUT_WINDOW_WIDTH);
	vY = glutGet(GLUT_WINDOW_HEIGHT);
	glViewport(0, 0, vX, vY);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glOrtho(0.0, X, 0.0, Y, -50.0, 50.0);
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
}

// glut initialization
void glut_initialize() {

	int myargc = 1;					
	char* myargv[1];
	myargv[0] = strdup("generate_network_network");

	glutInit(&myargc, myargv);									
	glutInitDisplayMode(GLUT_DEPTH | GLUT_DOUBLE | GLUT_RGBA);	
	glutInitWindowPosition(800, 0);							
	glutInitWindowSize(1000, 1000);								
	glutCreateWindow("Generate Simple 2D network");					

	glutDisplayFunc(glut_render);
	glutMouseFunc(glut_mouse);
	glutPassiveMotionFunc(glut_motion);
	glutMouseWheelFunc(glut_wheel);
	glutKeyboardFunc(glut_keyboard);
	glutReshapeFunc(glut_reshape);
	
	// initilize menu
	glutCreateMenu(glut_menu);					// create a menu object 
	glut_set_menu(0, 2);
	glutAttachMenu(GLUT_RIGHT_BUTTON);			// register right mouse to open menu option
}

// output an advertisement for the lab, authors and usage information
void advertise() {
	std::cout << std::endl << std::endl;
	std::cout << " =======================================================================================" << std::endl;
	std::cout << "|Thank you for using the synthetic microvascular model generator for microfluidics tool!|" << std::endl;
	std::cout << "|Scalable Tissue Imaging and Modeling (STIM) Lab, University of Houston                 |" << std::endl;
	std::cout << "|Developers: Jiaming Guo, David Mayerich                                                |" << std::endl;
	std::cout << "|Source: https://git.stim.ee.uh.edu/instrumentation/Microfluidics                       |" << std::endl;
	std::cout << " =======================================================================================" << std::endl << std::endl;

	std::cout << "usage: flow2" << std::endl;
	std::cout << "--2d -> activate 2d mode to treat the cross-section as rectangular" << std::endl;
	std::cout << "--units units-> string indicating output units (ex. um)" << std::endl;
	std::cout << "--maxpress 2 -> maximal pressure for simulation" << std::endl;
	std::cout << "--minradii 10 -> minimal manufacuture radius" << std::endl;
	std::cout << "--fradii 15 -> main feeder radius" << std::endl;
	std::cout << "--viscosity 0.00001 -> constant viscosity value" << std::endl;
	std::cout << "--workspace 450 -> workspace size in terms of units" << std::endl;
	std::cout << "--stackres 0.6 0.6 1.0 -> voxel size" << std::endl;
	std::cout << "--stackdir /home/network/image_stack -> image stack saving directory" << std::endl;
}

// argument and main loop
int main(int argc, char* argv[]) {
	
	HWND Window = GetConsoleWindow();									// set the window default window
	SetWindowPos(Window, 0, 0, 200, 0, 0, SWP_NOSIZE | SWP_NOZORDER);	// position might value based on the screen resolution

	stim::arglist args;													// create an instance of arglist

	// add arguments
	args.add("help", "prints this help");
	args.add("2d", "activate 2d mode and set the height of microvascular channel (in units), default is 3d mode (circle cross section)");
	args.add("units", "string indicating units of length for output measurements (ex. velocity)", "um", "text string");
	args.add("maxpress", "maximum allowed pressure in g / units / s^2, default 2 is for blood when units = um", "2", "real value > 0");
	args.add("minradii", "minimum radii allowed for manufacture, default 5 is for blood when units = um", "5", "real value > 5");
	args.add("fradii", "radii of main feeders, default is 10 when units = um", "10", "real value > 5");
	args.add("viscosity", "set the viscosity of the fluid (in g / units / s), default .00001 is for blood when units = um", ".00001", "real value > 0");
	args.add("workspace", "sets the size of the workspace (in units)", "400", "real value > 0");
	args.add("stackres", "spacing between pixel samples in each dimension(in units/pixel)", ".184 .184 1", "real value > 0");
	args.add("stackdir", "set the directory of the output image stack", "", "any existing directory (ex. /home/name/network)");

	args.parse(argc, argv);						// parse the command line

	// set up initial inputs
	if (args["help"].is_set()) {				// test for help
		advertise();							// advertise here
		std::cout << args.str();				// output arguments
		std::exit(1);
	}

	// get the units to work on
	units = args["units"].as_string();

	// set the mode, default is 10 in um
	if (args["2d"].is_set()) {
		mode = 2;
		h = args["2d"].as_float();
	}
	else {										// default mode is 3d
		mode = 3;
	}

	// get the workspace size
	X = Y = args["workspace"].as_float();

	// get the vexel and image stack size
	dx = args["stackres"].as_float(0);
	dy = args["stackres"].as_float(1);
	dz = args["stackres"].as_float(2);

	// get the save directory of image stack
	if (args["stackdir"].is_set())
		stackdir = args["stackdir"].as_string();

	// blood pressure in capillaries range from 15 - 35 torr
	// 1 torr = 133.3 Pa
	max_pressure = args["maxpress"].as_float();

	// normal blood viscosity range from 4 - 15 mPa·s(cP)
	// 1 Pa·s = 1 g / mm / s
	u = args["viscosity"].as_float();			// g / units / s

	// get minimum radii for building bridge
	default_radii = minimum_radii = args["minradii"].as_float();
	new_vertex.r = default_radii;

	// get the main feeder radius
	main_feeder_radii = args["fradii"].as_float();

	// draw a network					
	generate_network = true;				// begin draw a new network
	std::cout << " ==================" << std::endl;
	std::cout << "|  GENERATOR MODE  |" << std::endl;
	std::cout << " ==================" << std::endl << std::endl;
	std::cout << "[  TIP  ]  ";
	std::cout << "Click to draw a new network. (press SPACE to start a new line of edges)" << std::endl;
	std::cout << "[  NOTE  ] ";
	std::cout << "Press s to save the network and r to load the save" << std::endl;	

	// glut main loop
	glut_initialize();
	glutMainLoop();
}