#ifndef FLOW3_H
#define FLOW3_H

#include <algorithm>

//STIM include
#include <stim/parser/arguments.h>
#include <stim/visualization/gl_network.h>
#include <stim/visualization/colormap.h>
#include <stim/math/matrix.h>
#include <stim/visualization/gl_aaboundingbox.h>
#include <stim/ui/progressbar.h>
#include <stim/grids/image_stack.h>

#ifdef __CUDACC__
#include <cublas_v2.h>
#include <stim/cuda/cudatools/error.h>
#endif

namespace stim {
	template <typename A, typename B, typename C>
	struct triple {
		A first;
		B second;
		C third;
	};

	template <typename T>
	struct bridge {
		std::vector<unsigned> v;				// vertices' indices
		std::vector<typename stim::vec3<T> > V;	// vertices' coordinates
		T l;		// length
		T r;		// radius
		T deltaP;	// pressure drop
		T Q;		// volume flow rate
	};

	template <typename T>
	struct sphere {
		stim::vec3<T> c;		// center of sphere
		T r;					// radius
	};

	template <typename T>
	struct cone {				// radius changes gradually
		stim::vec3<T> c1;		// center of geometry start hat
		stim::vec3<T> c2;		// center of geometry end hat
		T r1;					// radius at start hat
		T r2;					// radius at end hat
	};

	template <typename T>
	struct cuboid {
		stim::vec3<T> c;
		T l;					// length
		T w;					// width
		T h;					// height
	};

	template <typename T>
	struct circuit {
		std::vector<typename std::pair<unsigned, unsigned> > v;		// end vertex index
		std::vector<T> r;											// branch resistence
	};

	/// indicator function
#ifdef __CUDACC__
	// for sphere
	template <typename T>
	__global__ void inside_sphere(const stim::sphere<T> *V, size_t num, T Z,size_t *R, T *S, unsigned char *ptr, int x, int y, int z, T std = 1.0f) {

		unsigned ix = blockDim.x * blockIdx.x + threadIdx.x;
		unsigned iy = blockDim.y * blockIdx.y + threadIdx.y;

		if (ix >= R[1] || iy >= R[2]) return;		// avoid seg-fault

		// find world_pixel coordinates
		stim::vec3<T> world_pixel;
		world_pixel[0] = (T)ix * S[1] - x;			// translate origin to center of the network
		world_pixel[1] = (T)iy * S[2] - y;
		world_pixel[2] = ((T)z - Z / 2.0f) * S[3];	// ???center of box minus half width

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
		else {
			T g = gaussianFunction(std::pow(distance - V[idx].r, 2), std);
			if(g > 0.1f)
				ptr[(R[2] - 1 - iy) * R[0] * R[1] + ix * R[0]] = 255;
		}
	}

	// for cone
	template <typename T>
	__global__ void inside_cone(const stim::cone<T> *E, size_t num, T Z, size_t *R, T *S, unsigned char *ptr, int x, int y, int z, T std = 1.0f) {

		unsigned ix = blockDim.x * blockIdx.x + threadIdx.x;
		unsigned iy = blockDim.y * blockIdx.y + threadIdx.y;

		if (ix >= R[1] || iy >= R[2]) return;			// avoid segfault

		stim::vec3<T> world_pixel;
		world_pixel[0] = (T)ix * S[1] - x;
		world_pixel[1] = (T)iy * S[2] - y;
		world_pixel[2] = ((T)z - Z / 2.0f) * S[3];

		float distance = FLT_MAX;
		float tmp_distance;
		float rr;										// radius at the surface where projection meets

		for (unsigned i = 0; i < num; i++) {			// find the nearest cylinder
			tmp_distance = ((world_pixel - E[i].c1).cross(world_pixel - E[i].c2)).len() / (E[i].c2 - E[i].c1).len();
			if (tmp_distance <= distance) {
				// we only focus on point to line segment
				// check to see whether projection is lying outside the line segment
				float a = (world_pixel - E[i].c1).dot((E[i].c2 - E[i].c1).norm());
				float b = (world_pixel - E[i].c2).dot((E[i].c1 - E[i].c2).norm());
				float length = (E[i].c1 - E[i].c2).len();
				if (a <= length && b <= length) {		// projection lying inside the line segment
					distance = tmp_distance;
					rr = E[i].r1 + (E[i].r2 - E[i].r1) * a / (length);		// linear change
				}
			}
		}
		if (distance <= rr)
			ptr[(R[2] - 1 - iy) * R[0] * R[1] + ix * R[0]] = 255;
		else {
			T g = gaussianFunction(std::pow(distance - rr, 2), std);
			if (g > 0.1f)
				ptr[(R[2] - 1 - iy) * R[0] * R[1] + ix * R[0]] = 255;
		}
	}

	// for source bus
	template <typename T>
	__global__ void inside_cuboid(const stim::cuboid<T> *B, size_t num, T Z, size_t *R, T *S, unsigned char *ptr, int x, int y, int z) {

		unsigned ix = blockDim.x * blockIdx.x + threadIdx.x;
		unsigned iy = blockDim.y * blockIdx.y + threadIdx.y;

		if (ix >= R[1] || iy >= R[2]) return;			// avoid segfault

		stim::vec3<T> world_pixel;
		world_pixel[0] = (T)ix * S[1] - x;
		world_pixel[1] = (T)iy * S[2] - y;
		world_pixel[2] = ((T)z - Z / 2.0f) * S[3];

		for (unsigned i = 0; i < num; i++) {
			bool left_outside = false;					// flag indicates point is outside the left bound
			bool right_outside = false;

			stim::vec3<T> tmp = B[i].c;
			stim::vec3<T> L = stim::vec3<T>(tmp[0] - B[i].l / 2.0f, tmp[1] - B[i].h / 2.0f, tmp[2] - B[i].w / 2.0f);
			stim::vec3<T> U = stim::vec3<T>(tmp[0] + B[i].l / 2.0f, tmp[1] + B[i].h / 2.0f, tmp[2] + B[i].w / 2.0f);

			for (unsigned d = 0; d < 3; d++) {
				if (world_pixel[d] < L[d])				// if the point is less than the minimum bound
					left_outside = true;
				if (world_pixel[d] > U[d])				// if the point is greater than the maximum bound
					right_outside = true;
			}
			if (!left_outside && !right_outside)
				ptr[(R[2] - 1 - iy) * R[0] * R[1] + ix * R[0]] = 255;
		}
	}
#endif

	template <typename T>
	class flow : public stim::gl_network<T> {

	private:

		unsigned num_edge;
		unsigned num_vertex;
		GLuint dlist;					// display list for inlets/outlets connections

		enum direction { UP, LEFT, DOWN, RIGHT };

		// calculate the cofactor of elemen[row][col]
		void get_minor(T** src, T** dest, int row, int col, int order) {

			// index of element to be copied
			int rowCount = 0;
			int colCount = 0;

			for (int i = 0; i < order; i++) {
				if (i != row) {
					colCount = 0;
					for (int j = 0; j < order; j++) {
						// when j is not the element
						if (j != col) {
							dest[rowCount][colCount] = src[i][j];
							colCount++;
						}
					}
					rowCount++;
				}
			}
		}

		// calculate the det()
		T determinant(T** mat, int order) {

			// degenate case when n = 1
			if (order == 1)
				return mat[0][0];

			T det = 0.0;		// determinant value

								// allocate the cofactor matrix
			T** minor = (T**)malloc((order - 1) * sizeof(T*));
			for (int i = 0; i < order - 1; i++)
				minor[i] = (T*)malloc((order - 1) * sizeof(T));


			for (int i = 0; i < order; i++) {

				// get minor of element(0, i)
				get_minor(mat, minor, 0, i, order);

				// recursion
				det += (i % 2 == 1 ? -1.0 : 1.0) * mat[0][i] * determinant(minor, order - 1);
			}

			// release memory
			for (int i = 0; i < order - 1; i++)
				free(minor[i]);
			free(minor);

			return det;
		}

	protected:

		using stim::network<T>::E;
		using stim::network<T>::V;
		using stim::network<T>::get_start_vertex;
		using stim::network<T>::get_end_vertex;
		using stim::network<T>::get_r;
		using stim::network<T>::get_average_r;
		using stim::network<T>::get_l;

		T** C;																	// Conductance
		std::vector<typename stim::triple<unsigned, unsigned, float> > Q;		// volume flow rate
		std::vector<T> QQ;														// Q' vector
		std::vector<T> pressure;												// final pressure
		std::vector<typename std::vector<typename stim::vec3<T> > > in_backup;	// inlet connection back up
		std::vector<typename std::vector<typename stim::vec3<T> > > out_backup;
		std::string units;														// length units

	public:

		bool set = false;														// flag indicates the pressure has been set
		std::vector<T> P;														// initial pressure
		std::vector<T> v;														// average velocity along each edge
		std::vector<typename stim::vec3<T> > main_feeder;						// inlet/outlet main feeder
		std::vector<unsigned> pendant_vertex;
		std::vector<typename stim::triple<unsigned, unsigned, T> > input;		// first one store which vertex, second one stores which edge, third one stores in/out volume flow rate of that vertex
		std::vector<typename stim::triple<unsigned, unsigned, T> > output;
		std::vector<typename stim::bridge<T> > inlet;							// input bridge
		std::vector<typename stim::bridge<T> > outlet;							// output bridge
		std::vector<typename stim::sphere<T> > A;			// sphere model for making image stack
		std::vector<typename stim::cone<T> > B;				// cone(cylinder) model for making image stack
		std::vector<typename stim::cuboid<T> > CU;			// cuboid model for making image stack
		stim::gl_aaboundingbox<T> bb;						// bounding box
		std::vector<bool> inlet_feasibility;				// list of flags indicate whether one inlet connection is feasible
		std::vector<bool> outlet_feasibility;
		std::vector<typename std::pair<stim::vec3<T>, stim::vec3<T> > > inbb;	// inlet connection bounding box
		std::vector<typename std::pair<stim::vec3<T>, stim::vec3<T> > > outbb;	// outlet connection bounding box
		T Ps;												// source and end pressure
		T Pe;

		flow() {}				// default constructor
		~flow() {
			for (unsigned i = 0; i < num_vertex; i++)
				delete[] C[i];
			delete[] C;
		}		// default destructor

		void init(unsigned n_e, unsigned n_v) {

			num_edge = n_e;
			num_vertex = n_v;

			C = new T*[n_v]();
			for (unsigned i = 0; i < n_v; i++) {
				C[i] = new T[n_v]();
			}

			QQ.resize(n_v);
			P.resize(n_v);
			pressure.resize(n_v);

			Q.resize(n_e);
			v.resize(n_e);
		}

		void clear() {

			for (unsigned i = 0; i < num_vertex; i++) {
				QQ[i] = 0;
				pressure[i] = 0;
				for (unsigned j = 0; j < num_vertex; j++) {
					C[i][j] = 0;
				}
			}
			main_feeder.clear();
			input.clear();
			output.clear();
			inlet.clear();
			outlet.clear();

			if (glIsList(dlist)) {
				glDeleteLists(dlist, 1);					// delete display list for modify
				glDeleteLists(dlist + 1, 1);
			}
		}

		void set_units(std::string u) {
			units = u;
		}

		// copy radius from cylinder to flow
		void set_radius(unsigned i, T radius) {

			for (unsigned j = 0; j < num_edge; j++) {
				if (E[j].v[0] == i)
					E[j].cylinder<T>::set_r(0, radius);
				else if (E[j].v[1] == i)
					E[j].cylinder<T>::set_r(E[j].size() - 1, radius);
			}
		}

		// get the radius of vertex i
		T get_radius(unsigned i) {

			unsigned tmp_e;				// edge index
			unsigned tmp_v;				// vertex index in that edge
			for (unsigned j = 0; j < num_edge; j++) {
				if (E[j].v[0] == i) {
					tmp_e = j;
					tmp_v = 0;
				}
				else if (E[j].v[1] == i) {
					tmp_e = j;
					tmp_v = (unsigned)(E[j].size() - 1);
				}
			}

			return E[tmp_e].r(tmp_v);
		}

		// get the radius of point j on edge i
		T get_radius(unsigned i, unsigned j) {
			return E[i].r(j);
		}

		// back up vertices
		std::vector<stim::vec3<T> > back_vertex() {
			std::vector<stim::vec3<T> > result;

			for (unsigned i = 0; i < num_vertex; i++)
				result.push_back(stim::vec3<T>(V[i][0], V[i][1], V[i][2]));

			return result;
		}

		// get pendant vertices
		std::vector<unsigned> get_pendant_vertex() {
			std::vector<unsigned> result;
			int count = 0;

			for (unsigned i = 0; i < V.size(); i++) {			// for every vertex
				for (unsigned j = 0; j < E.size(); j++) {		// for every edge
					if (i == E[j].v[0] || i == E[j].v[1])		// check whether current vertex terminates one edge
						count++;
				}
				if (count == 1) 								// is pendant vertex
					result.push_back(i);
				count = 0;										// reset count
			}

			return result;
		}

		// get the velocity of pendant vertex i
		T get_velocity(unsigned i) {

			unsigned tmp_e;				// edge index
			for (unsigned j = 0; j < num_edge; j++) {
				if (E[j].v[0] == i) {
					tmp_e = j;
					break;
				}
				else if (E[j].v[1] == i) {
					tmp_e = j;
					break;
				}
			}

			return v[tmp_e];
		}

		// set pressure at specifi vertex
		void set_pressure(unsigned i, T value) {
			P[i] = value;
		}

		// extrct the largest connected component
		void extract_lcc() {
			

		}

		// solve the linear system to get stable flow state
		void solve_flow(T viscosity) {

			// clear up last time simulation
			clear();

			// get the pendant vertex indices
			pendant_vertex = get_pendant_vertex();

			// get bounding box
			bb = (*this).boundingbox();

			// set the conductance matrix of flow object
			unsigned start_vertex = 0;
			unsigned end_vertex = 0;
			for (unsigned i = 0; i < num_edge; i++) {
				start_vertex = (unsigned)get_start_vertex(i);		// get the start vertex index of current edge
				end_vertex = (unsigned)get_end_vertex(i);			// get the end vertex index of current edge

				C[start_vertex][end_vertex] = -((T)stim::PI * std::pow(get_average_r(i), 4)) / (8 * u * get_l(i));

				C[end_vertex][start_vertex] = C[start_vertex][end_vertex];
			}
			// set the diagonal to the negative sum of row element
			float sum = 0.0;
			for (unsigned i = 0; i < num_vertex; i++) {
				for (unsigned j = 0; j < num_vertex; j++) {
					sum += C[i][j];
				}
				C[i][i] = -sum;
				sum = 0.0;
			}

			// get the Q' vector QQ
			// matrix manipulation to zero out the conductance matrix as defined by the boundary values that were enterd
			for (unsigned i = 0; i < num_vertex; i++) {
				if (P[i] != 0) {			// for every dangle vertex
					for (unsigned j = 0; j < num_vertex; j++) {
						if (j == i) {
							QQ[i] = C[i][i] * P[i];
						}
						else {
							C[i][j] = 0;
							QQ[j] = QQ[j] - C[j][i] * P[i];
							C[j][i] = 0;
						}
					}
				}
			}

			// get the inverse of conductance matrix
			stim::matrix<float> _C(num_vertex, num_vertex);
			inversion(C, num_vertex, _C.data());

			// get the pressure in the network
			for (unsigned i = 0; i < num_vertex; i++) {
				for (unsigned j = 0; j < num_vertex; j++) {
					pressure[i] += _C(i, j) * QQ[j];
				}
			}

			// get the flow state from known pressure
			T start_pressure = 0.0;
			T end_pressure = 0.0;
			T deltaP = 0.0;
			for (unsigned i = 0; i < num_edge; i++) {
				start_vertex = (unsigned)get_start_vertex(i);
				end_vertex = (unsigned)get_end_vertex(i);
				start_pressure = pressure[start_vertex];		// get the start vertex pressure of current edge
				end_pressure = pressure[end_vertex];			// get the end vertex pressure of current edge
				deltaP = start_pressure - end_pressure;				// deltaP = Pa - Pb

				Q[i].first = start_vertex;
				Q[i].second = end_vertex;

				Q[i].third = ((T)stim::PI * std::pow(get_average_r(i), 4) * deltaP) / (8 * u * get_l(i));
				v[i] = Q[i].third / ((T)stim::PI * std::pow(get_average_r(i), 2));
			}
		}

		// get the brewer color map based on velocity
		void get_color_map(T& max_v, T& min_v, std::vector<unsigned char>& color, std::vector<unsigned> pendant_vertex) {

			size_t num_edge = Q.size();
			size_t num_vertex = QQ.size();

			// find the absolute maximum velocity and minimum velocity
			std::vector<float> abs_V(num_edge);
			for (unsigned i = 0; i < num_edge; i++) {
				abs_V[i] = std::fabsf(v[i]);
			}

			max_v = *std::max_element(abs_V.begin(), abs_V.end());
			min_v = *std::min_element(abs_V.begin(), abs_V.end());

			// get the color map based on velocity range along the network
			color.clear();
			if (pendant_vertex.size() == 2 && num_edge - num_vertex + 1 <= 0) 		// only one inlet and one outlet
				color.resize(num_edge * 3, (unsigned char)128);
			else {
				color.resize(num_edge * 3);
				stim::cpu2cpu<float>(&abs_V[0], &color[0], num_edge, min_v, max_v, stim::cmBrewer);
			}
		}

		// print flow
		void print_flow() {

			// show the pressure information in console box
			std::cout << "PRESSURE(g/" << units << "/s^2):" << std::endl;
			for (unsigned i = 0; i < num_vertex; i++) {
				std::cout << "[" << i << "] " << pressure[i] << std::endl;
			}
			// show the flow rate information in console box
			std::cout << "VOLUME FLOW RATE(" << units << "^3/s):" << std::endl;
			for (unsigned i = 0; i < num_edge; i++) {
				std::cout << "(" << Q[i].first << "," << Q[i].second << ")" << Q[i].third << std::endl;
			}
		}

		/// helper function
		// find hilbert curve order
		// @param: current direct length between two vertices
		// @param: desire length
		void find_hilbert_order(T l, T d, int &order) {

			bool flag = false;
			int o = 1;
			T tmp;					// temp of length
			while (!flag) {
				// convert from cartesian length to hilbert length
				// l -> l * (4 ^ order - 1)/(2 ^ order - 1)
				tmp = (T)(l * (std::pow(4, o) - 1) / (std::pow(2, o) - 1));
				if (tmp >= d)
					flag = true;
				else
					o++;
			}
			order = o;
		}

		// move hilbert curves
		void move(unsigned i, T *c, direction dir, T dl, int feeder, bool invert) {

			int cof = (invert) ? -1 : 1;

			switch (dir) {
			case UP:
				c[1] += dl;
				break;
			case LEFT:
				c[0] -= cof * dl;
				break;
			case DOWN:
				c[1] -= dl;
				break;
			case RIGHT:
				c[0] += cof * dl;
				break;
			}

			stim::vec3<T> tmp;
			for (unsigned i = 0; i < 3; i++)
				tmp[i] = c[i];

			if (feeder == 1)					// inlet main feeder
				inlet[i].V.push_back(tmp);
			else if (feeder == 0)				// outlet main feeder
				outlet[i].V.push_back(tmp);
		}

		// form hilbert curves
		void hilbert_curve(unsigned i, T *c, int order, T dl, int feeder, bool invert, direction dir = DOWN) {

			if (order == 1) {
				switch (dir) {
				case UP:
					move(i, c, DOWN, dl, feeder, invert);
					move(i, c, RIGHT, dl, feeder, invert);
					move(i, c, UP, dl, feeder, invert);
					break;
				case LEFT:
					move(i, c, RIGHT, dl, feeder, invert);
					move(i, c, DOWN, dl, feeder, invert);
					move(i, c, LEFT, dl, feeder, invert);
					break;
				case DOWN:
					move(i, c, UP, dl, feeder, invert);
					move(i, c, LEFT, dl, feeder, invert);
					move(i, c, DOWN, dl, feeder, invert);
					break;
				case RIGHT:
					move(i, c, LEFT, dl, feeder, invert);
					move(i, c, UP, dl, feeder, invert);
					move(i, c, RIGHT, dl, feeder, invert);
					break;
				}

			}
			else if (order > 1) {
				switch (dir) {
				case UP:
					hilbert_curve(i, c, order - 1, dl, feeder, invert, LEFT);
					move(i, c, DOWN, dl, feeder, invert);
					hilbert_curve(i, c, order - 1, dl, feeder, invert, UP);
					move(i, c, RIGHT, dl, feeder, invert);
					hilbert_curve(i, c, order - 1, dl, feeder, invert, UP);
					move(i, c, UP, dl, feeder, invert);
					hilbert_curve(i, c, order - 1, dl, feeder, invert, RIGHT);
					break;
				case LEFT:
					hilbert_curve(i, c, order - 1, dl, feeder, invert, UP);
					move(i, c, RIGHT, dl, feeder, invert);
					hilbert_curve(i, c, order - 1, dl, feeder, invert, LEFT);
					move(i, c, DOWN, dl, feeder, invert);
					hilbert_curve(i, c, order - 1, dl, feeder, invert, LEFT);
					move(i, c, LEFT, dl, feeder, invert);
					hilbert_curve(i, c, order - 1, dl, feeder, invert, DOWN);
					break;
				case DOWN:
					hilbert_curve(i, c, order - 1, dl, feeder, invert, RIGHT);
					move(i, c, UP, dl, feeder, invert);
					hilbert_curve(i, c, order - 1, dl, feeder, invert, DOWN);
					move(i, c, LEFT, dl, feeder, invert);
					hilbert_curve(i, c, order - 1, dl, feeder, invert, DOWN);
					move(i, c, DOWN, dl, feeder, invert);
					hilbert_curve(i, c, order - 1, dl, feeder, invert, LEFT);
					break;
				case RIGHT:
					hilbert_curve(i, c, order - 1, dl, feeder, invert, DOWN);
					move(i, c, LEFT, dl, feeder, invert);
					hilbert_curve(i, c, order - 1, dl, feeder, invert, RIGHT);
					move(i, c, UP, dl, feeder, invert);
					hilbert_curve(i, c, order - 1, dl, feeder, invert, RIGHT);
					move(i, c, RIGHT, dl, feeder, invert);
					hilbert_curve(i, c, order - 1, dl, feeder, invert, UP);
					break;
				}
			}
		}

		/// render function
		// find two envelope caps for two spheres
		// @param cp1, cp2: list of points on the cap
		// @param center1, center2: center point of cap
		// @param r1, r2: radius of cap
		void find_envelope(std::vector<typename stim::vec3<float> > &cp1, std::vector<typename stim::vec3<float> > &cp2, stim::vec3<float> center1, stim::vec3<float> center2, float r1, float r2, GLint subdivision) {

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
				t1[2] = t2[2] = center1[2];		// decide the specific plane to work on
				t3[2] = t4[2] = center2[2];

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
				a = dot / (r1 * 1) * r1;			// a = cos(alpha) * radius
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

		// draw solid sphere at every vertex
		void glSolidSphere(T max_pressure, GLint subdivision, T scale = 1.0f) {

			// waste?
			for (unsigned i = 0; i < num_edge; i++) {
				// draw the starting vertex
				if (P[E[i].v[0]] != 0) {
					stim::vec3<float> new_color;
					new_color[0] = (P[E[i].v[0]] / max_pressure) > 0.5f ? 1.0f : 2.0f * P[E[i].v[0]] / max_pressure;						// red
					new_color[1] = 0.0f;																									// green
					new_color[2] = (P[E[i].v[0]] / max_pressure) > 0.5f ? 1.0f - 2.0f * (P[E[i].v[0]] / max_pressure - 0.5f) : 1.0f;		// blue
					glColor3f(new_color[0], new_color[1], new_color[2]);

					glPushMatrix();
					glTranslatef(E[i][0][0], E[i][0][1], E[i][0][2]);
					glutSolidSphere(get_r(i, 0) * scale, subdivision, subdivision);
					glPopMatrix();
				}
				else {
					glEnable(GL_BLEND);											// enable color blend
					glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);			// set blend function
					glDisable(GL_DEPTH_TEST);
					glColor4f(0.7f, 0.7f, 0.7f, 0.7f);							// gray color
					glPushMatrix();
					glTranslatef(E[i][0][0], E[i][0][1], E[i][0][2]);
					glutSolidSphere(get_r(i, 0) * scale, subdivision, subdivision);
					glPopMatrix();
					glDisable(GL_BLEND);
					glEnable(GL_DEPTH_TEST);
				}

				// draw the ending vertex
				if (P[E[i].v[1]] != 0) {
					stim::vec3<float> new_color;
					new_color[0] = (P[E[i].v[1]] / max_pressure) > 0.5f ? 1.0f : 2.0f * P[E[i].v[1]] / max_pressure;						// red
					new_color[1] = 0.0f;																									// green
					new_color[2] = (P[E[i].v[1]] / max_pressure) > 0.5f ? 1.0f - 2.0f * (P[E[i].v[1]] / max_pressure - 0.5f) : 1.0f;		// blue
					glColor3f(new_color[0], new_color[1], new_color[2]);

					glPushMatrix();
					glTranslatef(E[i][E[i].size() - 1][0], E[i][E[i].size() - 1][1], E[i][E[i].size() - 1][2]);
					glutSolidSphere(get_r(i, (unsigned)(E[i].size() - 1)) * scale, subdivision, subdivision);
					glPopMatrix();
				}
				else {
					glEnable(GL_BLEND);											// enable color blend
					glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);			// set blend function
					glDisable(GL_DEPTH_TEST);
					glColor4f(0.7f, 0.7f, 0.7f, 0.7f);							// gray color
					glPushMatrix();
					glTranslatef(E[i][E[i].size() - 1][0], E[i][E[i].size() - 1][1], E[i][E[i].size() - 1][2]);
					glutSolidSphere(get_r(i, (unsigned)(E[i].size() - 1)) * scale, subdivision, subdivision);
					glPopMatrix();
					glDisable(GL_BLEND);
					glEnable(GL_DEPTH_TEST);
				}
			}
		}

		// draw edges as series of cylinders
		void glSolidCylinder(unsigned index, std::vector<unsigned char> color, GLint subdivision, T scale = 1.0f) {

			stim::vec3<float> tmp_d;
			stim::vec3<float> center1;
			stim::vec3<float> center2;
			stim::circle<float> tmp_c;
			float r1;
			float r2;
			std::vector<typename stim::vec3<float> > cp1(subdivision + 1);
			std::vector<typename stim::vec3<float> > cp2(subdivision + 1);
			for (unsigned i = 0; i < num_edge; i++) {							// for every edge
				if (i == index) {												// render in tranparency for direction indication
					glEnable(GL_BLEND);											// enable color blend
					glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);			// set blend function
					glDisable(GL_DEPTH_TEST);
					glColor4f((float)color[i * 3 + 0] / 255, (float)color[i * 3 + 1] / 255, (float)color[i * 3 + 2] / 255, 0.3f);
				}
				else 
					glColor3f((float)color[i * 3 + 0] / 255, (float)color[i * 3 + 1] / 255, (float)color[i * 3 + 2] / 255);

				for (unsigned j = 0; j < E[i].size() - 1; j++) {				// for every point on the edge
					center1 = E[i][j];
					center2 = E[i][j + 1];

					r1 = get_r(i, j) * scale;
					r2 = get_r(i, j + 1) * scale;

					//// calculate the envelope caps
					//find_envelope(cp1, cp2, center1, center2, r1, r2, subdivision);
					if (j == 0) {
						if (E[i].size() == 2)
							find_envelope(cp1, cp2, center1, center2, r1, r2, subdivision);
						else {
							tmp_d = center2 - center1;
							tmp_d = tmp_d.norm();
							tmp_c.rotate(tmp_d);
							stim::circle<float> c1(center1, r1, tmp_d, tmp_c.U);
							cp1 = c1.glpoints(subdivision);
							tmp_d = (E[i][j + 2] - center2) + (center2 - center1);
							tmp_d = tmp_d.norm();
							tmp_c.rotate(tmp_d);
							stim::circle<float> c2(center2, r2, tmp_d, tmp_c.U);
							cp2 = c2.glpoints(subdivision);
						}
					}
					else if (j == E[i].size() - 2) {
						tmp_d = (center2 - center1) + (center1 - E[i][j - 1]);
						tmp_d = tmp_d.norm();
						tmp_c.rotate(tmp_d);
						stim::circle<float> c1(center1, r1, tmp_d, tmp_c.U);
						cp1 = c1.glpoints(subdivision);
						tmp_d = center2 - center1;
						tmp_d = tmp_d.norm();
						tmp_c.rotate(tmp_d);
						stim::circle<float> c2(center2, r2, tmp_d, tmp_c.U);
						cp2 = c2.glpoints(subdivision);
					} 
					else {
						tmp_d = (center2 - center1) + (center1 - E[i][j - 1]);
						tmp_d = tmp_d.norm();
						tmp_c.rotate(tmp_d);
						stim::circle<float> c1(center1, r1, tmp_d, tmp_c.U);
						cp1 = c1.glpoints(subdivision);
						tmp_d = (E[i][j + 2] - center2) + (center2 - center1);
						tmp_d = tmp_d.norm();
						tmp_c.rotate(tmp_d);
						stim::circle<float> c2(center2, r2, tmp_d, tmp_c.U);
						cp2 = c2.glpoints(subdivision);
					}

					glBegin(GL_QUAD_STRIP);
					for (unsigned j = 0; j < cp1.size(); j++) {
						glVertex3f(cp1[j][0], cp1[j][1], cp1[j][2]);
						glVertex3f(cp2[j][0], cp2[j][1], cp2[j][2]);
					}
					glEnd();
				}
				if (i == index) {
					glDisable(GL_BLEND);
					glEnable(GL_DEPTH_TEST);
				}
					
			}
			glFlush();
		}

		// draw the flow direction as cone, the size of the cone depends on the length of that edge
		void glSolidCone(unsigned i, GLint subdivision, T scale = 1.0f, T threshold = 0.01f) {

			stim::vec3<T> tmp_d;									// direction
			stim::vec3<T> center;									// cone hat center
			stim::vec3<T> head;										// cone hat top
			stim::circle<T> tmp_c;
			T h;													// height base of the cone
			std::vector<typename stim::vec3<T> > cp;
			T radius;

			glColor3f(0.0f, 0.0f, 0.0f);							// lime color

			size_t index = E[i].size() / 2 - 1;
			tmp_d = E[i][index + 1] - E[i][index];
			h = tmp_d.len() / 3.0f;									// get the height base by factor 3
			tmp_d = tmp_d.norm();
			center = (E[i][index + 1] + E[i][index]) / 2;
			tmp_c.rotate(tmp_d);
			radius = (E[i].r((unsigned)(index + 1)) + E[i].r((unsigned)index)) / 2 * scale;
			radius = (T)((h / sqrt(3) < radius) ? h / sqrt(3) : radius);	// update radius
			if (v[i] > threshold)
				head = center + tmp_d * h;
			else if (v[i] < -threshold)
				head = center - tmp_d * h;
			
			stim::circle<float> c(center, radius, tmp_d, tmp_c.U);
			cp = c.glpoints(subdivision);

			if (v[i] > threshold || v[i] < -threshold) {
				glBegin(GL_TRIANGLE_FAN);
				glVertex3f(head[0], head[1], head[2]);
				for (unsigned k = 0; k < cp.size(); k++)
					glVertex3f(cp[k][0], cp[k][1], cp[k][2]);
				glEnd();
				glFlush();
			}
			// draw a cone for every edge to indicate 
			//for (unsigned j = 0; j < E[i].size() - 1; j++) {	// for every point on current edge
			//	tmp_d = E[i][j + 1] - E[i][j];
			//	tmp_d = tmp_d.norm();
			//	center = (E[i][j + 1] + E[i][j]) / 2;
			//	tmp_c.rotate(tmp_d);
			//	radius = (E[i].r(j + 1) + E[i].r(j)) / 2;
			//	if (v[i] > 0)									// if flow flows from j to j+1
			//		head = center + tmp_d * 2 * sqrt(3) * radius;
			//	else
			//		head = center - tmp_d * 2 * sqrt(3) * radius;

			//	stim::circle<float> c(center, radius, tmp_d, tmp_c.U);
			//	cp = c.glpoints(subdivision);

			//	glBegin(GL_TRIANGLE_FAN);
			//	glVertex3f(head[0], head[1], head[2]);
			//	for (unsigned k = 0; k < cp.size(); k++)
			//		glVertex3f(cp[k][0], cp[k][1], cp[k][2]);
			//	glEnd();
			//}
			//glFlush();
		}
		void glSolidCone(GLint subdivision, T scale = 1.0f, T threhold = 0.01f) {

			stim::vec3<T> tmp_d;									// direction
			stim::vec3<T> center;									// cone hat center
			stim::vec3<T> head;										// cone hat top
			stim::circle<T> tmp_c;
			std::vector<typename stim::vec3<T> > cp;
			T h;
			T radius;

			glColor3f(0.600f, 0.847f, 0.788f);
			// draw a cone for every edge to indicate 
			for (unsigned i = 0; i < num_edge; i++) {				// for every edge
				unsigned k1 = E[i].size() / 2 - 1;					// start and end index
				unsigned k2 = E[i].size() / 2;
				tmp_d = E[i][k2] - E[i][k1];
				h = tmp_d.len() / 3.0f;								// get the height base by factor 3
				tmp_d = tmp_d.norm();
				center = (E[i][k2] + E[i][k1]) / 2;
				tmp_c.rotate(tmp_d);
				radius = (E[i].r(k2) + E[i].r(k1)) / 2 * scale;
				radius = (h / sqrt(3) < radius) ? h / sqrt(3) : radius;	// update radius by height base if necessary
				if (v[i] > threhold)										// if flow flows from k1 to k2
					head = center + tmp_d * h;
				else if(v[i] < -threhold)
					head = center - tmp_d * h;
				stim::circle<float> c(center, radius, tmp_d, tmp_c.U);
				cp = c.glpoints(subdivision);

				if (v[i] > threhold || v[i] < -threhold) {
					glBegin(GL_TRIANGLE_FAN);
					glVertex3f(head[0], head[1], head[2]);
					for (unsigned k = 0; k < cp.size(); k++)
						glVertex3f(cp[k][0], cp[k][1], cp[k][2]);
					glEnd();
				}
				//for (unsigned j = 0; j < E[i].size() - 1; j++) {	// for every point on current edge
				//	tmp_d = E[i][j + 1] - E[i][j];
				//	tmp_d = tmp_d.norm();
				//	center = (E[i][j + 1] + E[i][j]) / 2;
				//	tmp_c.rotate(tmp_d);
				//	radius = (E[i].r(j + 1) + E[i].r(j)) / 2;
				//	if (v[i] > 0)									// if flow flows from j to j+1
				//		head = center + tmp_d * 2 * sqrt(3) * radius;
				//	else
				//		head = center - tmp_d * 2 * sqrt(3) * radius;

				//	stim::circle<float> c(center, radius, tmp_d, tmp_c.U);
				//	cp = c.glpoints(subdivision);

				//	glBegin(GL_TRIANGLE_FAN);
				//	glVertex3f(head[0], head[1], head[2]);
				//	for (unsigned k = 0; k < cp.size(); k++)
				//		glVertex3f(cp[k][0], cp[k][1], cp[k][2]);
				//	glEnd();
				//}
			}
			glFlush();
		}

		// draw main feeder as solid cube
		void glSolidCuboid(GLint subdivision, bool manufacture = false, T length = 40.0f, T height = 10.0f) {

			T width;
			stim::gl_aaboundingbox<T> BB = (*this).boundingbox();
			stim::vec3<T> L = BB.A;						// get the bottom left corner
			stim::vec3<T> U = BB.B;						// get the top right corner
			width = U[2] - L[2] + 10.0f;

			if (manufacture)
				glColor3f(0.0f, 0.0f, 0.0f);			// black color
			else
				glColor3f(0.5f, 0.5f, 0.5f);			// gray color
			for (unsigned i = 0; i < main_feeder.size(); i++) {
				// front face
				glBegin(GL_QUADS);
				glVertex3f(main_feeder[i][0] - length / 2, main_feeder[i][1] - height / 2, main_feeder[i][2] - width / 2);
				glVertex3f(main_feeder[i][0] + length / 2, main_feeder[i][1] - height / 2, main_feeder[i][2] - width / 2);
				glVertex3f(main_feeder[i][0] + length / 2, main_feeder[i][1] + height / 2, main_feeder[i][2] - width / 2);
				glVertex3f(main_feeder[i][0] - length / 2, main_feeder[i][1] + height / 2, main_feeder[i][2] - width / 2);
				glEnd();

				// back face
				glBegin(GL_QUADS);
				glVertex3f(main_feeder[i][0] - length / 2, main_feeder[i][1] - height / 2, main_feeder[i][2] + width / 2);
				glVertex3f(main_feeder[i][0] + length / 2, main_feeder[i][1] - height / 2, main_feeder[i][2] + width / 2);
				glVertex3f(main_feeder[i][0] + length / 2, main_feeder[i][1] + height / 2, main_feeder[i][2] + width / 2);
				glVertex3f(main_feeder[i][0] - length / 2, main_feeder[i][1] + height / 2, main_feeder[i][2] + width / 2);
				glEnd();

				// top face
				glBegin(GL_QUADS);
				glVertex3f(main_feeder[i][0] - length / 2, main_feeder[i][1] + height / 2, main_feeder[i][2] - width / 2);
				glVertex3f(main_feeder[i][0] + length / 2, main_feeder[i][1] + height / 2, main_feeder[i][2] - width / 2);
				glVertex3f(main_feeder[i][0] + length / 2, main_feeder[i][1] + height / 2, main_feeder[i][2] + width / 2);
				glVertex3f(main_feeder[i][0] - length / 2, main_feeder[i][1] + height / 2, main_feeder[i][2] + width / 2);
				glEnd();

				// bottom face
				glBegin(GL_QUADS);
				glVertex3f(main_feeder[i][0] - length / 2, main_feeder[i][1] - height / 2, main_feeder[i][2] - width / 2);
				glVertex3f(main_feeder[i][0] + length / 2, main_feeder[i][1] - height / 2, main_feeder[i][2] - width / 2);
				glVertex3f(main_feeder[i][0] + length / 2, main_feeder[i][1] - height / 2, main_feeder[i][2] + width / 2);
				glVertex3f(main_feeder[i][0] - length / 2, main_feeder[i][1] - height / 2, main_feeder[i][2] + width / 2);
				glEnd();

				// left face
				glBegin(GL_QUADS);
				glVertex3f(main_feeder[i][0] - length / 2, main_feeder[i][1] - height / 2, main_feeder[i][2] - width / 2);
				glVertex3f(main_feeder[i][0] - length / 2, main_feeder[i][1] - height / 2, main_feeder[i][2] + width / 2);
				glVertex3f(main_feeder[i][0] - length / 2, main_feeder[i][1] + height / 2, main_feeder[i][2] + width / 2);
				glVertex3f(main_feeder[i][0] - length / 2, main_feeder[i][1] + height / 2, main_feeder[i][2] - width / 2);
				glEnd();

				// right face
				glBegin(GL_QUADS);
				glVertex3f(main_feeder[i][0] + length / 2, main_feeder[i][1] - height / 2, main_feeder[i][2] - width / 2);
				glVertex3f(main_feeder[i][0] + length / 2, main_feeder[i][1] + height / 2, main_feeder[i][2] - width / 2);
				glVertex3f(main_feeder[i][0] + length / 2, main_feeder[i][1] + height / 2, main_feeder[i][2] + width / 2);
				glVertex3f(main_feeder[i][0] + length / 2, main_feeder[i][1] - height / 2, main_feeder[i][2] + width / 2);
				glEnd();
			}
		}

		// draw flow velocity field, glyph
		void glyph(std::vector<unsigned char> color, GLint subdivision, T scale = 1.0f, bool frame = false, T r = 4.0f, T threshold = 0.01f) {
			
			// v1----v2-->v3
			T k = 4.0f;							// quartering
			stim::vec3<T> v1, v2, v3;			// three point
			stim::vec3<T> d;					// direction vector
			stim::circle<float> tmp_c;
			std::vector<typename stim::vec3<float> > cp1(subdivision + 1);
			std::vector<typename stim::vec3<float> > cp2(subdivision + 1);

			// rendering the arrows
			for (unsigned i = 0; i < num_edge; i++) {				// for every edge
				glColor3f((float)color[i * 3 + 0] / 255.0f, (float)color[i * 3 + 1] / 255.0f, (float)color[i * 3 + 2] / 255.0f);
				for (unsigned j = 0; j < E[i].size() - 1; j++) {	// for every point on that edge

					// consider the velocity valuence
					if (v[i] > threshold) {			// positive, from start point to end point
						v1 = E[i][j];
						v3 = E[i][j + 1];
					}
					else if (v[i] < -threshold) {		// negative, from end point to start point
						v1 = E[i][j + 1];
						v3 = E[i][j];
					}

					if (v[i] > threshold || v[i] < -threshold) {
						d = v3 - v1;
						// place the arrow in the middel of one edge
						v2 = v1 + (1.0f / k * 2.0f) * d;			// looks like =->=
						v1 = v1 + (1.0f / k) * d;
						v3 = v3 - (1.0f / k) * d;
						d = d.norm();
						tmp_c.rotate(d);

						// render the cylinder part
						stim::circle<T> c1(v1, r / 2 * scale, d, tmp_c.U);
						cp1 = c1.glpoints(subdivision);
						stim::circle<T> c2(v2, r / 2 * scale, d, tmp_c.U);
						cp2 = c2.glpoints(subdivision);

						glBegin(GL_QUAD_STRIP);
						for (unsigned k = 0; k < cp1.size(); k++) {
							glVertex3f(cp1[k][0], cp1[k][1], cp1[k][2]);
							glVertex3f(cp2[k][0], cp2[k][1], cp2[k][2]);
						}
						glEnd();

						// render the cone part
						stim::circle<T> c3(v2, r * scale, d, tmp_c.U);
						cp2 = c3.glpoints(subdivision);
						glBegin(GL_TRIANGLE_FAN);
						glVertex3f(v3[0], v3[1], v3[2]);
						for (unsigned k = 0; k < cp2.size(); k++)
							glVertex3f(cp2[k][0], cp2[k][1], cp2[k][2]);
						glEnd();
					}
				}
			}

			// rendering frames
			if (frame) {
				frame = false;
				stim::vec3<float> center1;
				stim::vec3<float> center2;
				stim::vec3<float> tmp_d;			// flow direction
				float r1, r2;

				for (unsigned i = 0; i < num_edge; i++) {
					for (unsigned j = 0; j < E[i].size() - 1; j++) {
						center1 = E[i][j];
						center2 = E[i][j + 1];

						r1 = get_r(i, j) * scale;
						r2 = get_r(i, j + 1) * scale;

						subdivision = 5;									// rough frames

						if (j == 0) {
							if (E[i].size() == 2)
								find_envelope(cp1, cp2, center1, center2, r1, r2, subdivision);
							else {
								tmp_d = center2 - center1;
								tmp_d = tmp_d.norm();
								tmp_c.rotate(tmp_d);
								stim::circle<float> c1(center1, r1, tmp_d, tmp_c.U);
								cp1 = c1.glpoints(subdivision);
								tmp_d = (E[i][j + 2] - center2) + (center2 - center1);
								tmp_d = tmp_d.norm();
								tmp_c.rotate(tmp_d);
								stim::circle<float> c2(center2, r2, tmp_d, tmp_c.U);
								cp2 = c2.glpoints(subdivision);
							}
						}
						else if (j == E[i].size() - 2) {
							tmp_d = (center2 - center1) + (center1 - E[i][j - 1]);
							tmp_d = tmp_d.norm();
							tmp_c.rotate(tmp_d);
							stim::circle<float> c1(center1, r1, tmp_d, tmp_c.U);
							cp1 = c1.glpoints(subdivision);
							tmp_d = center2 - center1;
							tmp_d = tmp_d.norm();
							tmp_c.rotate(tmp_d);
							stim::circle<float> c2(center2, r2, tmp_d, tmp_c.U);
							cp2 = c2.glpoints(subdivision);
						}
						else {
							tmp_d = (center2 - center1) + (center1 - E[i][j - 1]);
							tmp_d = tmp_d.norm();
							tmp_c.rotate(tmp_d);
							stim::circle<float> c1(center1, r1, tmp_d, tmp_c.U);
							cp1 = c1.glpoints(subdivision);
							tmp_d = (E[i][j + 2] - center2) + (center2 - center1);
							tmp_d = tmp_d.norm();
							tmp_c.rotate(tmp_d);
							stim::circle<float> c2(center2, r2, tmp_d, tmp_c.U);
							cp2 = c2.glpoints(subdivision);
						}

						glColor3f(140/255.0f, 81/255.0f, 10/255.0f);
						glBegin(GL_LINES);
						for (unsigned k = 0; k < cp1.size(); k++) {
							glVertex3f(cp1[k][0], cp1[k][1], cp1[k][2]);
							glVertex3f(cp2[k][0], cp2[k][1], cp2[k][2]);
						}
						glEnd();
					}
				}
			}
		}

		// display the total volume flow rate
		void display_flow_rate(T in, T out) {
			
			glMatrixMode(GL_PROJECTION);									// set up the 2d viewport for mode text printing
			glPushMatrix();
			glLoadIdentity();
			int X = glutGet(GLUT_WINDOW_WIDTH);								// get the current window width
			int Y = glutGet(GLUT_WINDOW_HEIGHT);							// get the current window height
			glViewport(0, 0, X, Y);											// locate to left bottom corner
			gluOrtho2D(0, X, 0, Y);											// define othogonal aspect
			glColor3f(0.8f, 0.0f, 0.0f);									// using red to show mode
			glMatrixMode(GL_MODELVIEW);
			glPushMatrix();
			glLoadIdentity();

			glRasterPos2f((GLfloat)(X / 2), (GLfloat)5.0f);					// hard coded position!!!!!
			std::stringstream ss_p;
			ss_p << "Q = ";				// Q = * um^3/s
			ss_p << in;					
			ss_p << " ";
			ss_p << units;
			ss_p << "^3/s";
			glutBitmapString(GLUT_BITMAP_TIMES_ROMAN_24, (const unsigned char*)(ss_p.str().c_str()));

			glPopMatrix();
			glMatrixMode(GL_PROJECTION);
			glPopMatrix();
		}

		// draw the bridge as lines or arrows
		void line_bridge(bool &redisplay, T r = 4.0f) {

			if (redisplay) {							// check to see whether the display list needs to be updated
				glDeleteLists(dlist, 1);
				redisplay = false;
			}

			if (!glIsList(dlist)) {
				dlist = glGenLists(1);
				glNewList(dlist, GL_COMPILE);

				//// render flow direction arrows
				//if (arrow) {
				//	// v1----v2-->v3
				//	T k = 4.0f;							// quartering
				//	stim::vec3<T> v1, v2, v3;			// three point
				//	stim::vec3<T> d;					// direction vector
				//	stim::circle<float> tmp_c;
				//	std::vector<typename stim::vec3<float> > cp1(subdivision + 1);
				//	std::vector<typename stim::vec3<float> > cp2(subdivision + 1);

				//	// inlet, right-going
				//	for (unsigned i = 0; i < inlet.size(); i++) {
				//		if (inlet_feasibility[i])
				//			glColor3f(0.0f, 0.0f, 0.0f);			// feasible -> black
				//		else
				//			glColor3f(1.0f, 0.0f, 0.0f);			// nonfeasible -> red
				//		for (unsigned j = 0; j < inlet[i].V.size() - 1; j++) {
				//			v1 = inlet[i].V[j];
				//			v3 = inlet[i].V[j + 1];
				//			d = v3 - v1;
				//			// place the arrow in the middel of one edge
				//			v2 = v1 + (1.0f / k * 2.0f) * d;			// looks like =->=
				//			v1 = v1 + (1.0f / k) * d;
				//			v3 = v3 - (1.0f / k) * d;
				//			d = d.norm();
				//			tmp_c.rotate(d);

				//			// render the cylinder part
				//			stim::circle<T> c1(v1, r / 2, d, tmp_c.U);
				//			cp1 = c1.glpoints(subdivision);
				//			stim::circle<T> c2(v2, r / 2, d, tmp_c.U);
				//			cp2 = c2.glpoints(subdivision);

				//			glBegin(GL_QUAD_STRIP);
				//			for (unsigned k = 0; k < cp1.size(); k++) {
				//				glVertex3f(cp1[k][0], cp1[k][1], cp1[k][2]);
				//				glVertex3f(cp2[k][0], cp2[k][1], cp2[k][2]);
				//			}
				//			glEnd();

				//			// render the cone part
				//			stim::circle<T> c3(v2, r, d, tmp_c.U);
				//			cp2 = c3.glpoints(subdivision);
				//			glBegin(GL_TRIANGLE_FAN);
				//			glVertex3f(v3[0], v3[1], v3[2]);
				//			for (unsigned k = 0; k < cp2.size(); k++)
				//				glVertex3f(cp2[k][0], cp2[k][1], cp2[k][2]);
				//			glEnd();
				//		}
				//	}

				//	// outlet, right-going
				//	for (unsigned i = 0; i < outlet.size(); i++) {
				//		if (outlet_feasibility[i])
				//			glColor3f(0.0f, 0.0f, 0.0f);			// feasible -> black
				//		else
				//			glColor3f(1.0f, 0.0f, 0.0f);			// nonfeasible -> red
				//		for (unsigned j = 0; j < outlet[i].V.size() - 1; j++) {
				//			v1 = outlet[i].V[j + 1];
				//			v3 = outlet[i].V[j];
				//			d = v3 - v1;
				//			// place the arrow in the middel of one edge
				//			v2 = v1 + (1.0f / k * 2.0f) * d;			// looks like =->=
				//			v1 = v1 + (1.0f / k) * d;
				//			v3 = v3 - (1.0f / k) * d;
				//			d = d.norm();
				//			tmp_c.rotate(d);

				//			// render the cylinder part
				//			stim::circle<T> c1(v1, r / 2, d, tmp_c.U);
				//			cp1 = c1.glpoints(subdivision);
				//			stim::circle<T> c2(v2, r / 2, d, tmp_c.U);
				//			cp2 = c2.glpoints(subdivision);

				//			glBegin(GL_QUAD_STRIP);
				//			for (unsigned k = 0; k < cp1.size(); k++) {
				//				glVertex3f(cp1[k][0], cp1[k][1], cp1[k][2]);
				//				glVertex3f(cp2[k][0], cp2[k][1], cp2[k][2]);
				//			}
				//			glEnd();

				//			// render the cone part
				//			stim::circle<T> c3(v2, r, d, tmp_c.U);
				//			cp2 = c3.glpoints(subdivision);
				//			glBegin(GL_TRIANGLE_FAN);
				//			glVertex3f(v3[0], v3[1], v3[2]);
				//			for (unsigned k = 0; k < cp2.size(); k++)
				//				glVertex3f(cp2[k][0], cp2[k][1], cp2[k][2]);
				//			glEnd();
				//		}
				//	}

				//	// render transparent lines as indexing
				//	glEnable(GL_BLEND);											// enable color blend
				//	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);			// set blend function
				//	glDisable(GL_DEPTH_TEST);
					glLineWidth(5);
					for (unsigned i = 0; i < inlet.size(); i++) {
						if (inlet_feasibility[i])
							glColor3f(0.0f, 0.0f, 0.0f);
				//			glColor4f(0.0f, 0.0f, 0.0f, 0.2f);		// feasible -> black
						else
							glColor3f(1.0f, 0.0f, 0.0f);
				//			glColor4f(1.0f, 0.0f, 0.0f, 0.2f);		// nonfeasible -> red

						glBegin(GL_LINE_STRIP);
						for (unsigned j = 0; j < inlet[i].V.size(); j++)
							glVertex3f(inlet[i].V[j][0], inlet[i].V[j][1], inlet[i].V[j][2]);
						glEnd();
					}
					for (unsigned i = 0; i < outlet.size(); i++) {
						if (outlet_feasibility[i])
							glColor3f(0.0f, 0.0f, 0.0f);
				//			glColor4f(0.0f, 0.0f, 0.0f, 0.2f);		// feasible -> black
						else
							glColor3f(1.0f, 0.0f, 0.0f);
				//			glColor4f(1.0f, 0.0f, 0.0f, 0.2f);		// nonfeasible -> red
						glBegin(GL_LINE_STRIP);
						for (unsigned j = 0; j < outlet[i].V.size(); j++)
							glVertex3f(outlet[i].V[j][0], outlet[i].V[j][1], outlet[i].V[j][2]);
						glEnd();
					}
				//	glDisable(GL_BLEND);
				//	glEnable(GL_DEPTH_TEST);
				//}
				glEndList();
			}
			glCallList(dlist);
		}

		// draw the bridge as tubes
		void tube_bridge(bool &redisplay, GLint subdivision, T scale = 1.0f, T radius = 5.0f) {

			if (redisplay) {
				glDeleteLists(dlist + 1, 1);
				redisplay = false;
			}

			if (!glIsList(dlist + 1)) {
				glNewList(dlist + 1, GL_COMPILE);

				stim::vec3<T> dir;							// direction vector
				stim::circle<T> unit_c;						// unit circle for finding the rotation start direction
				std::vector<typename stim::vec3<T> > cp1;
				std::vector<typename stim::vec3<T> > cp2;
				glColor3f(0.0f, 0.0f, 0.0f);

				for (unsigned i = 0; i < inlet.size(); i++) {
					// render vertex as sphere
					for (unsigned j = 1; j < inlet[i].V.size() - 1; j++) {
						glPushMatrix();
						glTranslatef(inlet[i].V[j][0], inlet[i].V[j][1], inlet[i].V[j][2]);
						glutSolidSphere(radius * scale, subdivision, subdivision);
						glPopMatrix();
					}
					// render edge as cylinder
					for (unsigned j = 0; j < inlet[i].V.size() - 1; j++) {
						dir = inlet[i].V[j] - inlet[i].V[j + 1];
						dir = dir.norm();
						unit_c.rotate(dir);
						stim::circle<T> c1(inlet[i].V[j], inlet[i].r * scale, dir, unit_c.U);
						stim::circle<T> c2(inlet[i].V[j + 1], inlet[i].r * scale, dir, unit_c.U);
						cp1 = c1.glpoints(subdivision);
						cp2 = c2.glpoints(subdivision);

						glBegin(GL_QUAD_STRIP);
						for (unsigned k = 0; k < cp1.size(); k++) {
							glVertex3f(cp1[k][0], cp1[k][1], cp1[k][2]);
							glVertex3f(cp2[k][0], cp2[k][1], cp2[k][2]);
						}
						glEnd();
					}
				}

				for (unsigned i = 0; i < outlet.size(); i++) {
					// render vertex as sphere
					for (unsigned j = 1; j < outlet[i].V.size() - 1; j++) {
						glPushMatrix();
						glTranslatef(outlet[i].V[j][0], outlet[i].V[j][1], outlet[i].V[j][2]);
						glutSolidSphere(radius * scale, subdivision, subdivision);
						glPopMatrix();
					}
					// render edge as cylinder
					for (unsigned j = 0; j < outlet[i].V.size() - 1; j++) {
						dir = outlet[i].V[j] - outlet[i].V[j + 1];
						dir = dir.norm();
						unit_c.rotate(dir);
						stim::circle<T> c1(outlet[i].V[j], outlet[i].r * scale, dir, unit_c.U);
						stim::circle<T> c2(outlet[i].V[j + 1], outlet[i].r * scale, dir, unit_c.U);
						cp1 = c1.glpoints(subdivision);
						cp2 = c2.glpoints(subdivision);

						glBegin(GL_QUAD_STRIP);
						for (unsigned k = 0; k < cp1.size(); k++) {
							glVertex3f(cp1[k][0], cp1[k][1], cp1[k][2]);
							glVertex3f(cp2[k][0], cp2[k][1], cp2[k][2]);
						}
						glEnd();
					}
				}
				glEndList();
			}
			glCallList(dlist + 1);
		}	

		// draw gradient color bounding box outside the object
		void bounding_box() {

			stim::vec3<T> L = bb.A;						// get the bottom left corner
			stim::vec3<T> U = bb.B;						// get the top right corner
			
			glLineWidth(1);
			// front face of the box (in L[2])
			glBegin(GL_LINE_LOOP);
			glColor3f(0.0f, 0.0f, 0.0f);
			glVertex3f(L[0], L[1], L[2]);
			glColor3f(0.0f, 1.0f, 0.0f);
			glVertex3f(L[0], U[1], L[2]);
			glColor3f(1.0f, 1.0f, 0.0f);
			glVertex3f(U[0], U[1], L[2]);
			glColor3f(1.0f, 0.0f, 0.0f);
			glVertex3f(U[0], L[1], L[2]);
			glEnd();

			// back face of the box (in U[2])
			glBegin(GL_LINE_LOOP);
			glColor3f(1.0f, 1.0f, 1.0f);
			glVertex3f(U[0], U[1], U[2]);
			glColor3f(0.0f, 1.0f, 1.0f);
			glVertex3f(L[0], U[1], U[2]);
			glColor3f(0.0f, 0.0f, 1.0f);
			glVertex3f(L[0], L[1], U[2]);
			glColor3f(1.0f, 0.0f, 1.0f);
			glVertex3f(U[0], L[1], U[2]);
			glEnd();

			// fill out the rest of the lines to connect the two faces
			glBegin(GL_LINES);
			glColor3f(0.0f, 1.0f, 0.0f);
			glVertex3f(L[0], U[1], L[2]);
			glColor3f(0.0f, 1.0f, 1.0f);
			glVertex3f(L[0], U[1], U[2]);
			glColor3f(1.0f, 1.0f, 1.0f);
			glVertex3f(U[0], U[1], U[2]);
			glColor3f(1.0f, 1.0f, 0.0f);
			glVertex3f(U[0], U[1], L[2]);
			glColor3f(1.0f, 0.0f, 0.0f);
			glVertex3f(U[0], L[1], L[2]);
			glColor3f(1.0f, 0.0f, 1.0f);
			glVertex3f(U[0], L[1], U[2]);
			glColor3f(0.0f, 0.0f, 1.0f);
			glVertex3f(L[0], L[1], U[2]);
			glColor3f(0.0f, 0.0f, 0.0f);
			glVertex3f(L[0], L[1], L[2]);
			glEnd();
		}

		// mark the vertex
		void mark_vertex(std::vector<stim::vec3<T> > vertex, T scale = 1.0f) {
			
			glColor3f(0.0f, 0.0f, 0.0f);
			for (unsigned i = 0; i < vertex.size(); i++) {
				glRasterPos3f(vertex[i][0], vertex[i][1] + get_radius(i) * scale, vertex[i][2]);		// place position
				std::stringstream ss;
				ss << i;																				// store index value
				glutBitmapString(GLUT_BITMAP_HELVETICA_18, (const unsigned char*)(ss.str().c_str()));
			}
		}

		// mark the edge
		void mark_edge() {

			glColor3f(0.0f, 1.0f, 0.0f);
			for (unsigned i = 0; i < num_edge; i++) {
				glRasterPos3f((V[E[i].v[0]] + V[E[i].v[1]])[0]/2, (V[E[i].v[0]] + V[E[i].v[1]])[1] / 2, (V[E[i].v[0]] + V[E[i].v[1]])[2] / 2);
				std::stringstream ss;
				ss << i;
				glutBitmapString(GLUT_BITMAP_HELVETICA_18, (const unsigned char*)(ss.str().c_str()));
			}
		}

		// find the nearest vertex of current click position
		// return true and a value if found
		inline bool epsilon_vertex(T x, T y, T z, T eps, T scale, unsigned& v) {

			T d = FLT_MAX;										// minimum distance between 2 vertices
			T tmp_d = 0.0f;										// temporary stores distance for loop
			unsigned tmp_i = 0;									// temporary stores connection index for loop
			stim::vec3<T> tmp_v;								// temporary stores current loop point
			d = FLT_MAX;										// set to max of float number

			for (unsigned i = 0; i < V.size(); i++) {
				tmp_v = stim::vec3<T>(x, y, z);
	
				tmp_v = tmp_v - V[i];							// calculate a vector between two vertices
				tmp_d = tmp_v.len();							// calculate length of that vector
				if (tmp_d < d) {
					d = tmp_d;									// if found a nearer vertex 
					tmp_i = i;									// get the index of that vertex
				}
			}
			eps += get_radius(tmp_i) * scale;					// increase epsilon accordingly
			if (d < eps) {										// if current click is close to any vertex
				v = tmp_i;										// copy the extant vertex's index to v
				return true;
			}

			return false;
		}

		// find the nearest inlet/outlet connection line of current click position
		// ab -> line segment, v -> point
		// return true and a value if found
		inline bool epsilon_edge(T x, T y, T z, T eps, unsigned &idx) {

			T d = FLT_MAX;
			T tmp_d;
			unsigned tmp_i = 0;
			unsigned tmp_j = 0;
			stim::vec3<T> v1;
			stim::vec3<T> v2;
			stim::vec3<T> v3;
			stim::vec3<T> v0 = stim::vec3<float>(x, y, z);
			bool online = false;					// flag indicates the point is on the line-segment
			float a, b;

			// inner network
			for (unsigned i = 0; i < E.size(); i++) {
				for (unsigned j = 0; j < E[i].size() - 1; j++) {
					v1 = E[i][j + 1] - E[i][j];		// a -> b = ab
					v2 = v0 - E[i][j];				// a -> v = av
					v3 = v0 - E[i][j + 1];			// b -> v = bv

					tmp_d = v2.dot(v1);				// av·ab

					// check the line relative position
					a = v2.dot(v1.norm());
					b = v3.dot(v1.norm());
					if (a < v1.len() && b < v1.len())		// if the length of projection fragment is longer than the line-segment
						online = true;
					else
						online = false;

					if (tmp_d <= 0.0 || tmp_d >= std::pow(v1.len(), 2) && !online)	// projection lies outside the line-segment
						continue;
					else {
						tmp_d = v1.cross(v2).len() / v1.len();						// perpendicular distance of point to segment: |v1 x v2| / |v1|
						if (tmp_d < d) {
							d = tmp_d;
							tmp_i = i;
							tmp_j = j;
						}
					} 
				}
			}

			eps += get_radius(tmp_i, tmp_j);

			if (d < eps) {
				idx = tmp_i;
				return true;
			}

			return false;
		}
		inline bool epsilon_edge(T x, T y, T z, T eps, unsigned &idx, unsigned &port) {

			T d = FLT_MAX;
			T tmp_d;
			unsigned tmp_i = 0;
			stim::vec3<T> v1;
			stim::vec3<T> v2;
			stim::vec3<T> v3;
			stim::vec3<T> v0 = stim::vec3<float>(x, y, z);
			bool online = false;					// flag indicates the point is on the line-segment
			float a, b;

			// inlet connection
			for (unsigned i = 0; i < inlet.size(); i++) {
				for (unsigned j = 0; j < inlet[i].V.size() - 1; j++) {
					v1 = inlet[i].V[j + 1] - inlet[i].V[j];
					v2 = v0 - inlet[i].V[j];
					v3 = v0 - inlet[i].V[j + 1];

					tmp_d = v2.dot(v1);				// av·ab

					// check the line relative position
					a = v2.dot(v1.norm());
					b = v3.dot(v1.norm());
					if (a < v1.len() && b < v1.len())		// if the length of projection fragment is longer than the line-segment
						online = true;
					else
						online = false;

					if (tmp_d <= 0.0 || tmp_d > std::pow(v1.len(), 2) && !online)	// projection lies outside the line-segment
						continue;
					else {
						tmp_d = v1.cross(v2).len() / v1.len();						// perpendicular distance of point to segment: |v1 x v2| / |v1|
						if (tmp_d < d) {
							d = tmp_d;
							tmp_i = i;
							port = 0;
						}
					}
				
				}
			}

			// outlet connection
			for (unsigned i = 0; i < outlet.size(); i++) {
				for (unsigned j = 0; j < outlet[i].V.size() - 1; j++) {
					v1 = outlet[i].V[j + 1] - outlet[i].V[j];
					v2 = v0 - outlet[i].V[j];
					v3 = v0 - outlet[i].V[j + 1];

					tmp_d = v2.dot(v1);				// av·ab

					// check the line relative position
					a = v2.dot(v1.norm());
					b = v3.dot(v1.norm());
					if (a < v1.len() && b < v1.len())		// if the length of projection fragment is longer than the line-segment
						online = true;
					else
						online = false;

					if (tmp_d <= 0.0 || tmp_d > std::pow(v1.len(), 2) && !online)	// projection lies outside the line-segment
						continue;
					else {
						tmp_d = v1.cross(v2).len() / v1.len();						// perpendicular distance of point to segment: |v1 x v2| / |v1|
						if (tmp_d < d) {
							d = tmp_d;
							tmp_i = i;
							port = 1;
						}
					}
				}
			}

			if (d < eps) {
				idx = tmp_i;
				return true;
			}

			return false;
		}

		/// build main feeder connection
		// set up main feeder and main port of both input and output
		void set_main_feeder(T border = 120.0f) {
			
			// 0 means outgoing while 1 means incoming
			stim::vec3<T> inlet_main_feeder;
			stim::vec3<T> outlet_main_feeder;

			inlet_main_feeder = stim::vec3<T>(bb.A[0] - border, bb.center()[1], bb.center()[2]);
			outlet_main_feeder = stim::vec3<T>(bb.B[0] + border, bb.center()[1], bb.center()[2]);
			
			main_feeder.push_back(inlet_main_feeder);		// 0->inlet, 1->outlet
			main_feeder.push_back(outlet_main_feeder);

			// find both input and output vertex
			stim::triple<unsigned, unsigned, float> tmp;
			size_t N = pendant_vertex.size();				// get the number of dangle vertex
			unsigned idx = 0;
			for (unsigned i = 0; i < N; i++) {				// for every boundary vertex
				idx = pendant_vertex[i];
				for (unsigned j = 0; j < num_edge; j++) {	// for every edge
					if (Q[j].first == idx) {			// starting vertex
						if (Q[j].third > 0) {			// flow comes in
							tmp.first = idx;
							tmp.second = j;
							tmp.third = Q[j].third;
							input.push_back(tmp);
							break;
						}
						// their might be a degenerate case that it equals to 0?
						else if (Q[j].third < 0) {		// flow comes out
							tmp.first = idx;
							tmp.second = j;
							tmp.third = -Q[j].third;
							output.push_back(tmp);
							break;
						}
					}
					else if (Q[j].second == idx) {		// ending vertex
						if (Q[j].third > 0) {			// flow comes in
							tmp.first = idx;
							tmp.second = j;
							tmp.third = Q[j].third;
							output.push_back(tmp);
							break;
						}
						// their might be a degenerate case that it equals to 0?
						else if (Q[j].third < 0) {		// flow comes out
							tmp.first = idx;
							tmp.second = j;
							tmp.third = -Q[j].third;
							input.push_back(tmp);
							break;
						}
					}
				}
			}
		}

		// build connection between all inlets and outlets
		// connection will trail along one axis around the bounding box
		void build_synthetic_connection(T viscosity, T radius = 5.0f) {
			
			stim::vec3<T> L = bb.A;						// get the bottom left corner
			stim::vec3<T> U = bb.B;						// get the top right corner
			T box_length = U[0] - L[0];
			T x0, dx;

			stim::vec3<T> tmp_v;						// start vertex
			stim::vec3<T> mid_v;						// middle point of the bridge
			stim::vec3<T> bus_v;						// point on the bus
			x0 = main_feeder[0][0] + 15.0f;				// assume bus length is 40.0f
			for (unsigned i = 0; i < input.size(); i++) {
				
				tmp_v = V[input[i].first];
				dx = 30.0f * ((tmp_v[0] - L[0]) / box_length);		// the socket position depends on proximity
				bus_v = stim::vec3<T>(x0 - dx, main_feeder[0][1], tmp_v[2]);
				mid_v = stim::vec3<T>(x0 - dx, tmp_v[1], tmp_v[2]);

				stim::bridge<T> tmp_b;
				tmp_b.V.push_back(bus_v);
				tmp_b.V.push_back(mid_v);
				tmp_b.V.push_back(tmp_v);
				tmp_b.v.push_back(input[i].first);
				tmp_b.Q = input[i].third;
				tmp_b.l = (bus_v - mid_v).len() + (mid_v - tmp_v).len();
				tmp_b.r = radius;

				inlet.push_back(tmp_b);
			}

			x0 = main_feeder[1][0] - 15.0f;
			for (unsigned i = 0; i < output.size(); i++) {

				tmp_v = V[output[i].first];
				dx = 30.0f * ((U[0] - tmp_v[0]) / box_length);		// the socket position depends on proximity
				bus_v = stim::vec3<T>(x0 + dx, main_feeder[1][1], tmp_v[2]);
				mid_v = stim::vec3<T>(x0 + dx, tmp_v[1], tmp_v[2]);

				stim::bridge<T> tmp_b;
				tmp_b.V.push_back(bus_v);
				tmp_b.V.push_back(mid_v);
				tmp_b.V.push_back(tmp_v);
				tmp_b.v.push_back(output[i].first);
				tmp_b.Q = output[i].third;
				tmp_b.l = (bus_v - mid_v).len() + (mid_v - tmp_v).len();
				tmp_b.r = radius;

				outlet.push_back(tmp_b);
			}

			backup();
		}

		// find the number of U-shape or square-shape structure for extending length of connection
		// @param t: width = t * radius
		int find_number_square(T origin_l, T desire_l, int times = 10, T radius = 5.0f) {
			
			bool done = false;						// flag indicates the current number of square shape structure is feasible
			int n = (int)(origin_l / (times * 4 * radius));	// number of square shape structure
			T need_l = desire_l - origin_l;
			T height;								// height of the square shapce structure

			while (!done) {
				height = need_l / (2 * n);			// calculate the height
				if (height > 2 * radius) {
					done = true;
				}
				else {
					n--;
				}
			}
			
			return n;
		}

		// build square connections
		void build_square_connection(int i, T width, T height, T origin_l, T desire_l, int n, int feeder, T threshold, bool z, bool left = true, bool up = true, int times = 10, T ratio = 0, T radius = 5.0f) {
		
			int coef_up = (up) ? 1 : -1;				// y coefficient
			int coef_left = (left) ? 1 : -1;			// x coefficient
			int coef_z = (z) ? 1 : -1;					// z coefficient
			int inverse = 1;							// inverse flag
			stim::vec3<T> cor_v;						// corner vertex
			std::pair<stim::vec3<T>, stim::vec3<T>> tmp_bb;
			stim::vec3<T> tmp_v;
			if (feeder == 1) 
				tmp_v = inlet[i].V[inlet[i].V.size() - 1];
			else if (feeder == 0) 
				tmp_v = outlet[i].V[outlet[i].V.size() - 1];
			tmp_bb.first = tmp_v;

			// pre-set fragments
			if (ratio) {
				T tmp_d, tmp_l;														// back ups
				tmp_d = desire_l;	
				tmp_l = origin_l;

				cor_v = tmp_v + stim::vec3<T>(-coef_left * origin_l, 0, 0);			// get the original corner vertex
				desire_l = desire_l - origin_l * (1.0f - ratio / 1.0f);
				origin_l = (T)origin_l * ratio / 1.0f;
				n = find_number_square(origin_l, desire_l, times);
				
				width = (T)origin_l / (2 * n);										// updates
				height = (desire_l - origin_l) / (2 * n);
				
				// there are cases that the fragment can not satisfy the requirement for width
				if (width < times * radius || n == 0) {								// check feasibility
					ratio = 1.0f;													// load original lengths
					desire_l = tmp_d;												// loading back-ups										
					origin_l = tmp_l;

					std::cout << "Warning: current ratio is not feasible, use full original line." << std::endl;
					n = find_number_square(origin_l, desire_l, times);

					width = (T)origin_l / (2 * n);									// updates
					height = (desire_l - origin_l) / (2 * n);
				}
			}

			// check whether it needs 3D square-wave-like connections
			if (height > threshold) {					// enbale 3D connections
				
				height = (T)((desire_l - (1 + 2 * n) * origin_l) / std::pow(2 * n, 2));	// compute new height in 3D structure
				while (height > threshold) {			// increase order to decrease height
					n++;
					width = (T)(origin_l) / (2 * n);
					height = (T)((desire_l - (1 + 2 * n) * origin_l) / std::pow(2 * n, 2));
					// check whether it appears overlap, if it appears we choose last time height even if it is larger than threshold
					if (width < times * radius) {
						n--;
						width = (T)(origin_l) / (2 * n);
						height = (T)((desire_l - (1 + 2 * n) * origin_l) / std::pow(2 * n, 2));
						break;
					}
				}

				// check overlap in terms of height, has potential risk when both height and width are less than times * radius.
				while (height < times * radius) {
					n--;
					width = (T)(origin_l) / (2 * n);
					height = (T)((desire_l - (1 + 2 * n) * origin_l) / std::pow(2 * n, 2));
				}

				/// degenerated case, use original extending method
				if (n == 0) {
					height = (T)((desire_l - origin_l) / 2);
					// "up"
					tmp_v = tmp_v + stim::vec3<T>(0, coef_up * height, 0);
					if (feeder == 1)
						inlet[i].V.push_back(tmp_v);
					else if (feeder == 0)
						outlet[i].V.push_back(tmp_v);
					// "left"
					tmp_v = tmp_v + stim::vec3<T>(-coef_left * origin_l, 0, 0);
					if (feeder == 1)
						inlet[i].V.push_back(tmp_v);
					else if (feeder == 0)
						outlet[i].V.push_back(tmp_v);
				}
			
				// cube-like structure construction
				for (int j = 0; j < n; j++) {
					// "up"
					for (int k = 0; k < n; k++) {
						// in
						tmp_v = tmp_v + stim::vec3<T>(0, 0, coef_z * height);
						if (feeder == 1)
							inlet[i].V.push_back(tmp_v);
						else if (feeder == 0)
							outlet[i].V.push_back(tmp_v);
						// "up"
						tmp_v = tmp_v + stim::vec3<T>(0, inverse * coef_up * width, 0);
						if (feeder == 1)
							inlet[i].V.push_back(tmp_v);
						else if (feeder == 0)
							outlet[i].V.push_back(tmp_v);
						// out
						tmp_v = tmp_v + stim::vec3<T>(0, 0, -coef_z * height);
						if (feeder == 1)
							inlet[i].V.push_back(tmp_v);
						else if (feeder == 0)
							outlet[i].V.push_back(tmp_v);
						// "up"
						tmp_v = tmp_v + stim::vec3<T>(0, inverse * coef_up * width, 0);
						if (feeder == 1)
							inlet[i].V.push_back(tmp_v);
						else if (feeder == 0)
							outlet[i].V.push_back(tmp_v);
					}

					// "left"
					tmp_v = tmp_v + stim::vec3<T>(-coef_left * width, 0, 0);
					if (feeder == 1)
						inlet[i].V.push_back(tmp_v);
					else if (feeder == 0)
						outlet[i].V.push_back(tmp_v);

					if (inverse == 1)					// revert inverse
						inverse = -1;
					else
						inverse = 1;

					// "down"
					for (int k = 0; k < n; k++) {
						// in
						tmp_v = tmp_v + stim::vec3<T>(0, 0, coef_z * height);
						if (feeder == 1)
							inlet[i].V.push_back(tmp_v);
						else if (feeder == 0)
							outlet[i].V.push_back(tmp_v);
						// get the bounding box edge
						if (j == n - 1 && k == 0)		// first time go "in"
							tmp_bb.second = tmp_v;
						// "down"
						tmp_v = tmp_v + stim::vec3<T>(0, inverse * coef_up * width, 0);
						if (feeder == 1)
							inlet[i].V.push_back(tmp_v);
						else if (feeder == 0)
							outlet[i].V.push_back(tmp_v);
						// out
						tmp_v = tmp_v + stim::vec3<T>(0, 0, -coef_z * height);
						if (feeder == 1)
							inlet[i].V.push_back(tmp_v);
						else if (feeder == 0)
							outlet[i].V.push_back(tmp_v);
						// "down"
						tmp_v = tmp_v + stim::vec3<T>(0, inverse * coef_up * width, 0);
						if (feeder == 1)
							inlet[i].V.push_back(tmp_v);
						else if (feeder == 0)
							outlet[i].V.push_back(tmp_v);
					}

					// "left"
					tmp_v = tmp_v + stim::vec3<T>(-coef_left * width, 0, 0);
					if (feeder == 1)
						inlet[i].V.push_back(tmp_v);
					else if (feeder == 0)
						outlet[i].V.push_back(tmp_v);

					if (inverse == 1)					// revert inverse
						inverse = -1;
					else
						inverse = 1;
				}
				// if use fragment to do square wave connection, need to push_back the corner vertex
				if (ratio > 0.0f && ratio < 1.0f) {
					if (feeder == 1)
						inlet[i].V.push_back(cor_v);
					else if (feeder == 0)
						outlet[i].V.push_back(cor_v);
				}
			}
			// use 2D square-wave-like connections
			else {
				if (height < times * radius) {			// if height is too small, decrease n and re-calculate height and width
					height = times * radius;
					T need_l = desire_l - origin_l;
					n = (int)(need_l / (2 * height));
					if (n == 0)							// degenerated case
						n = 1;
					height = need_l / (2 * n);
					width = origin_l / (2 * n);
				}
				for (int j = 0; j < n; j++) {

					// up
					tmp_v = tmp_v + stim::vec3<T>(0, coef_up * height, 0);
					if (feeder == 1)
						inlet[i].V.push_back(tmp_v);
					else if (feeder == 0)
						outlet[i].V.push_back(tmp_v);

					// left
					tmp_v = tmp_v + stim::vec3<T>(-coef_left * width, 0, 0);
					if (feeder == 1)
						inlet[i].V.push_back(tmp_v);
					else if (feeder == 0)
						outlet[i].V.push_back(tmp_v);
					if (j == n - 1)
						tmp_bb.second = tmp_v;

					// down
					tmp_v = tmp_v + stim::vec3<T>(0, -coef_up * height, 0);
					if (feeder == 1)
						inlet[i].V.push_back(tmp_v);
					else if (feeder == 0)
						outlet[i].V.push_back(tmp_v);

					// left
					tmp_v = tmp_v + stim::vec3<T>(-coef_left * width, 0, 0);
					if (feeder == 1)
						inlet[i].V.push_back(tmp_v);
					else if (feeder == 0)
						outlet[i].V.push_back(tmp_v);
				}
				// if use fragment to do square wave connection, need to push_back the corner vertex
				if (ratio > 0.0f && ratio < 1.0f) {
					if (feeder == 1)
						inlet[i].V.push_back(cor_v);
					else if (feeder == 0)
						outlet[i].V.push_back(cor_v);
				}
			}
			if (feeder == 1)
				inbb[i] = tmp_bb;
			else if (feeder == 0)
				outbb[i] = tmp_bb;
		}

		// automatically modify bridge to make it feasible
		void modify_synthetic_connection(T viscosity, T rou, bool H, T threshold, T &in, T &out, int times = 10, T ratio = 0.0f, T radius = 5.0f) {

			glDeleteLists(dlist, 1);					// delete display list for modify
			glDeleteLists(dlist + 1, 1);
			
			// because of radius change at the port vertex, there will be a pressure drop at that port
			// it follows the bernoulli equation
			// p1 + 1/2*rou*v1^2 + rou*g*h1 = p2 + 1/2*rou*v2^2 + rou*g*h2
			// Q1 = Q2 -> v1*r1^2 = v2*r2^2
			std::vector<T> new_pressure = pressure;
			unsigned idx;
			for (unsigned i = 0; i < pendant_vertex.size(); i++) {
				idx = pendant_vertex[i];
				T tmp_v = get_velocity(idx);			// velocity at that pendant vertex
				T ar = get_radius(idx) / radius;
				new_pressure[idx] = pressure[idx] + 1.0f / 2.0f * rou * std::pow(tmp_v, 2) * (1.0f - std::pow(ar, 4));
			}

			// increase r -> increase Q -> decrease l
			// find maximum pressure inlet port
			T source_pressure = FLT_MIN;	// source pressure
			unsigned inlet_index;
			T tmp_p;
			for (unsigned i = 0; i < inlet.size(); i++) {
				tmp_p = new_pressure[inlet[i].v[0]] + ((8 * viscosity * inlet[i].l * inlet[i].Q) / ((float)stim::PI * std::pow(radius, 4)));
				if (tmp_p > source_pressure) {
					source_pressure = tmp_p;
					inlet_index = i;
				}
			}
			Ps = source_pressure;

			// find minimum pressure outlet port
			T end_pressure = FLT_MAX;
			unsigned outlet_index;
			for (unsigned i = 0; i < outlet.size(); i++) {
				tmp_p = new_pressure[outlet[i].v[0]] - ((8 * viscosity * outlet[i].l * outlet[i].Q) / ((float)stim::PI * std::pow(radius, 4)));
				if (tmp_p < end_pressure) {
					end_pressure = tmp_p;
					outlet_index = i;
				}
			}
			Pe = end_pressure;

			// automatically modify inlet bridge using Hilbert curves
			if (H) {
				bool upper = false;						// flag indicates the whether the port is upper than main feeder
				bool invert = false;					// there are two version of hilbert curve depends on starting position with respect to the cup
				T new_l;
				stim::vec3<T> bus_v;					// the port point on the bus
				stim::vec3<T> mid_v;					// the original corner point
				stim::vec3<T> tmp_v;					// the pendant point
				int order = 0;							// order of hilbert curve (iteration)
				for (unsigned i = 0; i < inlet.size(); i++) {
					if (i != inlet_index) {
						new_l = (source_pressure - new_pressure[inlet[i].v[0]]) * ((float)stim::PI * std::pow(radius, 4)) / (8 * viscosity * inlet[i].Q);

						if (inlet[i].V[2][1] > main_feeder[0][1]) {		// check out upper side of lower side
							upper = true;
							invert = false;
						}
						else {
							upper = false;
							invert = true;
						}

						T origin_l = (inlet[i].V[1] - inlet[i].V[2]).len();
						T desire_l = new_l - (inlet[i].V[0] - inlet[i].V[1]).len();
						find_hilbert_order(origin_l, desire_l, order);

						bus_v = inlet[i].V[0];
						mid_v = inlet[i].V[1];
						tmp_v = inlet[i].V[2];
						inlet[i].V.clear();
						inlet[i].V.push_back(tmp_v);
						inlet[i].l = new_l;

						if (desire_l - origin_l < 2 * radius) {	// do not need to use hilbert curve, just increase the length by draging out
							T d = new_l - inlet[i].l;
							stim::vec3<T> corner = stim::vec3<T>(tmp_v[0], tmp_v[1] + d / 2.0f * (tmp_v[1] > main_feeder[0][1] ? 1 : -1), tmp_v[2]);
							inlet[i].V.push_back(corner);
							corner = stim::vec3<T>(mid_v[0], mid_v[1] + d / 2.0f * (tmp_v[1] > main_feeder[0][1] ? 1 : -1), mid_v[2]);
							inlet[i].V.push_back(corner);
							inlet[i].V.push_back(bus_v);
						}
						else {
							T fragment = (T)((desire_l - origin_l) / ((std::pow(4, order) - 1) / (std::pow(2, order) - 1) - 1));	// the length of the opening of cup 		
							T dl = (T)(fragment / (std::pow(2, order) - 1));											// unit cup length

							if (dl > 2 * radius) {				// if the radius is feasible
								if (upper)
									hilbert_curve(i, &tmp_v[0], order, dl, 1, invert, DOWN);
								else
									hilbert_curve(i, &tmp_v[0], order, dl, 1, invert, UP);

								if (tmp_v[0] != mid_v[0])
									inlet[i].V.push_back(mid_v);
								inlet[i].V.push_back(bus_v);
							}
							else {								// if the radius is not feasible
								int count = 1;
								while (dl <= 2 * radius) {
									dl = (T)(origin_l / (std::pow(2, order - count) - 1));
									count++;
								}
								count--;

								if (upper)
									hilbert_curve(i, &tmp_v[0], order - count, dl, 1, invert, DOWN);
								else
									hilbert_curve(i, &tmp_v[0], order - count, dl, 1, invert, UP);

								desire_l -= (T)(origin_l * ((std::pow(4, order - count) - 1) / (std::pow(2, order - count) - 1)));
								origin_l = (bus_v - mid_v).len();
								desire_l += origin_l;

								find_hilbert_order(origin_l, desire_l, order);

								fragment = (T)((desire_l - origin_l) / ((std::pow(4, order) - 1) / (std::pow(2, order) - 1) - 1));
								dl = (T)(fragment / (std::pow(2, order) - 1));
								if (dl < 2 * radius)
									std::cout << "infeasible connection between inlets!" << std::endl;

								if (upper)
									hilbert_curve(i, &tmp_v[0], order, dl, 1, !invert, LEFT);
								else
									hilbert_curve(i, &tmp_v[0], order, dl, 1, !invert, RIGHT);

								if (tmp_v[1] != bus_v[1])
									inlet[i].V.push_back(bus_v);
							}
						}
						std::reverse(inlet[i].V.begin(), inlet[i].V.end());			// from bus to pendant vertex
					}
				}

				// automatically modify outlet bridge to make it feasible
				for (unsigned i = 0; i < outlet.size(); i++) {
					if (i != outlet_index) {
						new_l = (new_pressure[outlet[i].v[0]] - end_pressure) * ((float)stim::PI * std::pow(radius, 4)) / (8 * viscosity * outlet[i].Q);

						if (outlet[i].V[2][1] > main_feeder[1][1]) {
							upper = true;
							invert = true;
						}
						else {
							upper = false;
							invert = false;
						}

						T origin_l = (outlet[i].V[1] - outlet[i].V[2]).len();
						T desire_l = new_l - (outlet[i].V[0] - outlet[i].V[1]).len();
						find_hilbert_order(origin_l, desire_l, order);

						bus_v = outlet[i].V[0];
						mid_v = outlet[i].V[1];
						tmp_v = outlet[i].V[2];
						outlet[i].V.clear();
						outlet[i].V.push_back(tmp_v);
						outlet[i].l = new_l;

						if (desire_l - origin_l < 2 * radius) {	// do not need to use hilbert curve, just increase the length by draging out
							T d = new_l - outlet[i].l;
							stim::vec3<T> corner = stim::vec3<T>(tmp_v[0], tmp_v[1] + d / 2.0f * (tmp_v[1] > main_feeder[0][1] ? 1 : -1), tmp_v[2]);
							outlet[i].V.push_back(corner);
							corner = stim::vec3<T>(mid_v[0], mid_v[1] + d / 2.0f * (tmp_v[1] > main_feeder[0][1] ? 1 : -1), mid_v[2]);
							outlet[i].V.push_back(corner);
							outlet[i].V.push_back(bus_v);
						}
						else {
							T fragment = (T)((desire_l - origin_l) / ((std::pow(4, order) - 1) / (std::pow(2, order) - 1) - 1));	// the length of the opening of cup 		
							T dl = (T)(fragment / (std::pow(2, order) - 1));											// unit cup length

							if (dl > 2 * radius) {				// if the radius is feasible
								if (upper)
									hilbert_curve(i, &tmp_v[0], order, dl, 0, invert, DOWN);
								else
									hilbert_curve(i, &tmp_v[0], order, dl, 0, invert, UP);

								if (tmp_v[0] != mid_v[0])
									outlet[i].V.push_back(mid_v);
								outlet[i].V.push_back(bus_v);
							}
							else {								// if the radius is not feasible
								int count = 1;
								while (dl <= 2 * radius) {
									dl = (T)(origin_l / (std::pow(2, order - count) - 1));
									count++;
								}
								count--;

								if (upper)
									hilbert_curve(i, &tmp_v[0], order - count, dl, 0, invert, DOWN);
								else
									hilbert_curve(i, &tmp_v[0], order - count, dl, 0, invert, UP);

								desire_l -= (T)(origin_l * ((std::pow(4, order - count) - 1) / (std::pow(2, order - count) - 1)));
								origin_l = (bus_v - mid_v).len();
								desire_l += origin_l;

								find_hilbert_order(origin_l, desire_l, order);

								fragment = (T)((desire_l - origin_l) / ((std::pow(4, order) - 1) / (std::pow(2, order) - 1) - 1));
								dl = (T)(fragment / (std::pow(2, order) - 1));
								if (dl < 2 * radius)
									std::cout << "infeasible connection between outlets!" << std::endl;

								if (upper)
									hilbert_curve(i, &tmp_v[0], order, dl, 0, !invert, LEFT);
								else
									hilbert_curve(i, &tmp_v[0], order, dl, 0, !invert, RIGHT);

								if (tmp_v[1] != bus_v[1])
									outlet[i].V.push_back(bus_v);
							}
						}
						std::reverse(outlet[i].V.begin(), outlet[i].V.end());
					}
				}
			}
			// automatically modify inlet bridge using square shape constructions
			else {
				bool upper;								// flag indicates the connection is upper than the bus
				bool z;									// flag indicates the connection direction along z-axis
				T new_l;								// new length
				stim::vec3<T> bus_v;					// the port point on the bus
				stim::vec3<T> mid_v;					// the original corner point
				stim::vec3<T> tmp_v;					// the pendant point
				int n;
				T width, height;						// width and height of the square
				inbb.resize(inlet.size());				// resize bounding box of inlets/outlets connections
				outbb.resize(outlet.size());

				for (unsigned i = 0; i < inlet.size(); i++) {
					if (i != inlet_index) {
						new_l = (source_pressure - new_pressure[inlet[i].v[0]]) * ((float)stim::PI * std::pow(radius, 4)) / (8 * viscosity * inlet[i].Q);	// calculate the new length of the connection 

						bus_v = inlet[i].V[0];
						mid_v = inlet[i].V[1];
						tmp_v = inlet[i].V[2];										// not always pendant vertex

						if (inlet[i].V[2][1] > main_feeder[0][1]) 					// check out upper side of lower side
							upper = true;
						else
							upper = false;

						if (inlet[i].V[2][2] > main_feeder[0][2])
							z = true;
						else
							z = false;

						T origin_l = (inlet[i].V[1] - inlet[i].V[2]).len();
						T desire_l = new_l - (inlet[i].V[0] - inlet[i].V[1]).len();
						if (inlet[i].V.size() != 3) {
							desire_l = new_l - (inlet[i].V[0] - inlet[i].V[1]).len() - (inlet[i].V[2] - inlet[i].V[3]).len();
							stim::vec3<T> tmp = inlet[i].V[3];
							inlet[i].V.clear();
							inlet[i].V.push_back(tmp);
							inlet[i].V.push_back(tmp_v);
						}
						else {
							inlet[i].V.clear();
							inlet[i].V.push_back(tmp_v);
						}
						inlet[i].l = new_l;

						n = find_number_square(origin_l, desire_l, times);

						width = (T)origin_l / (2 * n);
						height = (desire_l - origin_l) / (2 * n);

						build_square_connection(i, width, height, origin_l, desire_l, n, 1, threshold, z, true, upper, 2, ratio);
						inlet[i].V.push_back(bus_v);

						std::reverse(inlet[i].V.begin(), inlet[i].V.end());			// from bus to pendant vertex
					}
					else {
						inbb[i].first = inlet[i].V[2];
						inbb[i].second = inlet[i].V[1];
					}
				}

				for (unsigned i = 0; i < outlet.size(); i++) {
					if (i != outlet_index) {
						new_l = (new_pressure[outlet[i].v[0]] - end_pressure) * ((float)stim::PI * std::pow(radius, 4)) / (8 * viscosity * outlet[i].Q);	// calculate the new length of the connection 

						bus_v = outlet[i].V[0];
						mid_v = outlet[i].V[1];
						tmp_v = outlet[i].V[2];

						if (outlet[i].V[2][1] > main_feeder[1][1]) 					// check out upper side of lower side
							upper = true;
						else
							upper = false;

						if (outlet[i].V[2][2] > main_feeder[1][2])
							z = true;
						else
							z = false;

						T origin_l = (outlet[i].V[1] - outlet[i].V[2]).len();
						T desire_l = new_l - (outlet[i].V[0] - outlet[i].V[1]).len();
						if (outlet[i].V.size() != 3) {
							desire_l = new_l - (outlet[i].V[0] - outlet[i].V[1]).len() - (outlet[i].V[2] - outlet[i].V[3]).len();
							stim::vec3<T> tmp = outlet[i].V[3];
							outlet[i].V.clear();
							outlet[i].V.push_back(tmp);
							outlet[i].V.push_back(tmp_v);
						}
						else {
							outlet[i].V.clear();
							outlet[i].V.push_back(tmp_v);
						}
						outlet[i].l = new_l;

						n = find_number_square(origin_l, desire_l, times);

						width = (T)origin_l / (2 * n);
						height = (desire_l - origin_l) / (2 * n);

						build_square_connection(i, width, height, origin_l, desire_l, n, 0, threshold, z, false, upper, 2, ratio);
						outlet[i].V.push_back(bus_v);

						std::reverse(outlet[i].V.begin(), outlet[i].V.end());			// from bus to pendant vertex
					}
					else {
						outbb[i].first = outlet[i].V[2];
						outbb[i].second = outlet[i].V[1];
					}
				}
			}

			// save in-/out- volume flow rate
			in = out = 0.0f;
			for (unsigned i = 0; i < inlet.size(); i++)
				in += inlet[i].Q;
			for (unsigned i = 0; i < outlet.size(); i++)
				out += outlet[i].Q;

			check_special_connection();				// check special connections
		}

		/// check current connections to find overlapping
		// phase 1 check -> direct connection intersection
		void check_direct_connection() {
			
			size_t num;
			// check inlet
			num = inlet.size();								// get the number of inlets
			inlet_feasibility.resize(num, true);			// initialization
			for (unsigned i = 0; i < num; i++) {
				for (unsigned j = 0; j < num; j++) {
					if (i != j) {
						if (inlet[i].V[0][1] == inlet[j].V[0][1]) {
							if ((inlet[i].V[1][0] >= inlet[j].V[1][0]) && (fabs(inlet[i].V[1][1]) >= fabs(inlet[j].V[1][1])) && (((inlet[i].V[1][1] - main_feeder[0][1]) * (inlet[j].V[1][1] - main_feeder[0][1])) > 0 ? 1 : 0) && inlet[i].V[1][2] == inlet[j].V[1][2]) {
								inlet_feasibility[i] = false;
								break;
							}
							else
								inlet_feasibility[i] = true;
						}
					}
				}
			}
			// check outlet
			num = outlet.size();
			outlet_feasibility.resize(num, true);
			for (unsigned i = 0; i < num; i++) {
				for (unsigned j = 0; j < num; j++) {
					if (i != j) {
						if (outlet[i].V[0][2] == outlet[j].V[0][2]) {
							if ((outlet[i].V[1][0] <= outlet[j].V[1][0]) && (fabs(outlet[i].V[1][1]) >= fabs(outlet[j].V[1][1])) && (((outlet[i].V[1][1] - main_feeder[1][1]) * (outlet[j].V[1][1] - main_feeder[1][1])) > 0 ? 1 : 0) && outlet[i].V[1][2] == outlet[j].V[1][2]) {
								outlet_feasibility[i] = false;
								break;
							}
						}
						else
							outlet_feasibility[i] = true;
					}
				}
			}
		}

		// phase 2 check -> special connection intersection
		void check_special_connection(T radius = 5.0f) {
		
			// temp AABB centers and halfwidths
			stim::vec3<T> c1, c2;
			stim::vec3<T> r1, r2;
			// inlets' special connections checking
			for (unsigned i = 0; i < inbb.size(); i++) {
				for (unsigned j = 0; j < inbb.size(); j++) {
					if (j != i) {
						c1 = stim::vec3<T>((inbb[i].first + inbb[i].second) / 2);
						c2 = stim::vec3<T>((inbb[j].first + inbb[j].second) / 2);
						for (unsigned k = 0; k < 3; k++) {
							r1[k] = fabs(inbb[i].first[k] - inbb[i].second[k]) / 2;
							r2[k] = fabs(inbb[j].first[k] - inbb[j].second[k]) / 2;
						}
						// test AABBAABB
						if (fabs(c1[0] - c2[0]) > (r1[0] + r2[0] + 2 * radius) || fabs(c1[1] - c2[1]) > (r1[1] + r2[1] + 2 * radius) || fabs(c1[2] - c2[2]) > (r1[2] + r2[2] + 2 * radius))
							inlet_feasibility[i] = true;
						else
							inlet_feasibility[i] = false;
					}
				}
			}

			// outlets' special connections checking
			for (unsigned i = 0; i < outbb.size(); i++) {
				for (unsigned j = 0; j < outbb.size(); j++) {
					if (j != i) {
						c1 = stim::vec3<T>((outbb[i].first + outbb[i].second) / 2);
						c2 = stim::vec3<T>((outbb[j].first + outbb[j].second) / 2);
						for (unsigned k = 0; k < 3; k++) {
							r1[k] = fabs(outbb[i].first[k] - outbb[i].second[k]) / 2;
							r2[k] = fabs(outbb[j].first[k] - outbb[j].second[k]) / 2;
						}
						// test AABBAABB
						if (fabs(c1[0] - c2[0]) > (r1[0] + r2[0] + 2 * radius) || fabs(c1[1] - c2[1]) > (r1[1] + r2[1] + 2 * radius) || fabs(c1[2] - c2[2]) > (r1[2] + r2[2] + 2 * radius))
							outlet_feasibility[i] = true;
						else
							outlet_feasibility[i] = false;
					}
				}
			}
		}
		
		// clear synthetic connections
		void clear_synthetic_connection() {
			
			// restore direct synthetic connecions
			T l = 0.0f;
			for (unsigned i = 0; i < inlet.size(); i++) {
				inlet[i].V.clear();
				for (unsigned j = 0; j < in_backup[i].size(); j++) {
					inlet[i].V.push_back(in_backup[i][j]);
					if (j != in_backup[i].size() - 1)
						l += (in_backup[i][j + 1] - in_backup[i][j]).len();
				}
				inlet[i].l = l;
				l = 0.0f;
			}
			for (unsigned i = 0; i < outlet.size(); i++) {
				outlet[i].V.clear();
				for (unsigned j = 0; j < out_backup[i].size(); j++) {
					outlet[i].V.push_back(out_backup[i][j]);
					if (j != out_backup[i].size() - 1)
						l += (out_backup[i][j + 1] - out_backup[i][j]).len();
				}
				outlet[i].l = l;
				l = 0.0f;
			}

			// clear up inlets/outlets connection bounding box
			inbb.clear();
			outbb.clear();
		}

		// back up direct synthetic connection whenever modified
		void backup() {
			
			in_backup.clear();
			out_backup.clear();

			// back up direct synthetic connecions
			std::vector<typename stim::vec3<T> > V;
			for (unsigned i = 0; i < inlet.size(); i++) {
				for (unsigned j = 0; j < inlet[i].V.size(); j++) {
					V.push_back(inlet[i].V[j]);
				}
				in_backup.push_back(V);
				V.clear();
			}
			for (unsigned i = 0; i < outlet.size(); i++) {
				for (unsigned j = 0; j < outlet[i].V.size(); j++) {
					V.push_back(outlet[i].V[j]);
				}
				out_backup.push_back(V);
				V.clear();
			}
		}

		/// adjustment in order to match microfluidics experiments
		void adjust(T in, T out, T &Rt, T nQ, T viscosity, T radius = 5.0f) {
			
			// compute total resistance
			Rt = (Ps - Pe) / in;
			Pe = 0.0f;

			Ps = Rt * nQ;

			// adjust synthetic connections velocity flow rate. (linear scale)
			T k = nQ / in;				// linear factor
			for (unsigned i = 0; i < inlet.size(); i++) {
				inlet[i].Q *= k;
				input[i].third *= k;
			}
			for (unsigned i = 0; i < outlet.size(); i++) {
				outlet[i].Q *= k;
				output[i].third *= k;
			}
				
			/// simulate inner network flow
			// clear up initialized pressure
			P.resize(num_vertex);
			for (unsigned i = 0; i < pendant_vertex.size(); i++) {
				unsigned index = UINT_MAX;
				for (unsigned j = 0; j < inlet.size(); j++) {
					if (inlet[j].v[0] == pendant_vertex[i]) {
						index = j;
						break;
					}
				}
				if (index != UINT_MAX) {
					P[inlet[index].v[0]] = (T)(Ps - (8 * viscosity * inlet[index].l * inlet[index].Q / (stim::PI * std::pow(radius, 4))));
				}
			}

			for (unsigned i = 0; i < pendant_vertex.size(); i++) {
				unsigned index = UINT_MAX;
				for (unsigned j = 0; j < outlet.size(); j++) {
					if (outlet[j].v[0] == pendant_vertex[i]) {
						index = j;
						break;
					}
				}
				if (index != UINT_MAX) {
					P[outlet[index].v[0]] = (T)(Pe + (8 * viscosity * outlet[index].l * outlet[index].Q / (stim::PI * std::pow(radius, 4))));
				}
			}

			// clear up previous simulation except synthetic connection parts
			for (unsigned i = 0; i < num_vertex; i++) {
				QQ[i] = 0;
				pressure[i] = 0;
				for (unsigned j = 0; j < num_vertex; j++) {
					C[i][j] = 0;
				}
			}

			// re-simulation
			solve_flow(viscosity);
		}

		/// make full-synthetic binary image stack
		// prepare for image stack
		void preparation(T &Xl, T &Xr, T &Yt, T &Yb, T &Z, bool prototype = false, T length = 40.0f, T height = 10.0f, T radius = 5.0f, T scale = 1.0f) {
			
			T max_radius = 0.0f;
			T top = FLT_MIN;
			T bottom = FLT_MAX;

			// clear up last time result
			A.clear();
			B.clear();
			CU.clear();

			// firstly push back the original network
			stim::sphere<T> new_sphere;
			stim::cone<T> new_cone;
			stim::cuboid<T> new_cuboid;

			// take every source bus as cuboid
			new_cuboid.c = main_feeder[0];
			new_cuboid.l = length;
			new_cuboid.w = bb.B[2] - bb.A[2] + 10.0f;
			new_cuboid.h = height;
			CU.push_back(new_cuboid);
			new_cuboid.c = main_feeder[1];
			CU.push_back(new_cuboid);

			// take every point as sphere, every line as cone
			if (!prototype) {
				for (unsigned i = 0; i < num_edge; i++) {
					for (unsigned j = 0; j < E[i].size(); j++) {
						new_sphere.c = E[i][j];
						new_sphere.r = E[i].r(j) * scale;
						A.push_back(new_sphere);
						if (j != E[i].size() - 1) {
							new_cone.c1 = E[i][j];
							new_cone.c2 = E[i][j + 1];
							new_cone.r1 = E[i].r(j) * scale;
							new_cone.r2 = E[i].r(j + 1) * scale;
							B.push_back(new_cone);
						}
					}
				}
			}
			
			// secondly push back outside connection
			for (unsigned i = 0; i < inlet.size(); i++) {
				for (unsigned j = 1; j < inlet[i].V.size() - 1; j++) {
					new_sphere.c = inlet[i].V[j];
					new_sphere.r = inlet[i].r * scale;
					A.push_back(new_sphere);
				}
			}
			for (unsigned i = 0; i < outlet.size(); i++) {
				for (unsigned j = 1; j < outlet[i].V.size() - 1; j++) {
					new_sphere.c = outlet[i].V[j];
					new_sphere.r = outlet[i].r * scale;
					A.push_back(new_sphere);
				}
			}

			for (unsigned i = 0; i < inlet.size(); i++) {
				for (unsigned j = 0; j < inlet[i].V.size() - 1; j++) {
					new_cone.c1 = inlet[i].V[j];
					new_cone.c2 = inlet[i].V[j + 1];
					new_cone.r1 = inlet[i].r * scale;
					new_cone.r2 = inlet[i].r * scale;
					B.push_back(new_cone);
				}
			}
			for (unsigned i = 0; i < outlet.size(); i++) {
				for (unsigned j = 0; j < outlet[i].V.size() - 1; j++) {
					new_cone.c1 = outlet[i].V[j];
					new_cone.c2 = outlet[i].V[j + 1];
					new_cone.r1 = outlet[i].r * scale;
					new_cone.r2 = outlet[i].r * scale;
					B.push_back(new_cone);
				}
			}

			// find out the image stack size
			Xl = main_feeder[0][0] - length / 2 - 2 * radius;			// left bound x coordinate
			Xr = main_feeder[1][0] + length / 2 + 2 * radius;			// right bound x coordinate

			for (unsigned i = 0; i < A.size(); i++) {
				if (A[i].c[1] > top)
					top = A[i].c[1];
				if (A[i].c[1] < bottom)
					bottom = A[i].c[1];
				// extend the network boundingbox if the additional connections are outside
				if (A[i].c[2] > bb.B[2])
					bb.B[2] = A[i].c[2];
				if (A[i].c[2] < bb.A[2])
					bb.A[2] = A[i].c[2];
				if (A[i].r * scale > max_radius)
					max_radius = A[i].r * scale;
			}

			Yt = top + 2 * radius;							// top bound y coordinate
			Yb = bottom - 2 * radius;						// bottom bound y coordinate
			Z = (bb.B[2] - bb.A[2] + 4 * radius);			// bounding box width(along z-axis)
		}

		/// making image stack main function
		void make_image_stack(stim::image_stack<unsigned char, T> &I, T dx, T dy, T dz, std::string stackdir, bool prototype = false, T radius = 5.0f, T scale = 1.0f) {
			
			/// preparation for making image stack
			T X, Xl, Xr, Y, Yt, Yb, Z;
			preparation(Xl, Xr, Yt, Yb, Z, prototype, 40.0f, 10.0f, radius, scale);
			X = Xr - Xl;								// bounding box length(along x-axis)
			Y = Yt - Yb;								// bounding box height(along y-axis)
			stim::vec3<T> center = bb.center();			// get the center of bounding box
			int size_x, size_y, size_z;

			if (!prototype) {
				/// make
				size_x = (int)(X / dx + 1);			// set the size of image
				size_y = (int)(Y / dy + 1);
				size_z = (int)(Z / dz + 1);			// +3 in order to deal with reminder
				///  initialize image stack object
				I.init(1, size_x, size_y, size_z);
				I.set_dim(dx, dy, dz);
			}
			else {
				size_x = (int)I.nx();
				size_y = (int)I.ny();
				size_z = (int)I.nz();
			}
			
			// because of lack of memory, we have to computer one slice of stack per time
			// allocate vertex, edge and bus
			stim::sphere<T> *d_V;
			stim::cone<T> *d_E;
			stim::cuboid<T> *d_B;

			HANDLE_ERROR(cudaMalloc((void**)&d_V, A.size() * sizeof(stim::sphere<T>)));
			HANDLE_ERROR(cudaMalloc((void**)&d_E, B.size() * sizeof(stim::cone<T>)));
			HANDLE_ERROR(cudaMalloc((void**)&d_B, CU.size() * sizeof(stim::cuboid<T>)));
			HANDLE_ERROR(cudaMemcpy(d_V, &A[0], A.size() * sizeof(stim::sphere<T>), cudaMemcpyHostToDevice));
			HANDLE_ERROR(cudaMemcpy(d_E, &B[0], B.size() * sizeof(stim::cone<T>), cudaMemcpyHostToDevice));
			HANDLE_ERROR(cudaMemcpy(d_B, &CU[0], CU.size() * sizeof(stim::cuboid<T>), cudaMemcpyHostToDevice));

			// allocate image stack information memory
			size_t* d_R;
			T *d_S;

			size_t* R = (size_t*)malloc(4 * sizeof(size_t));	// size in 4 dimension
			R[0] = 1;
			R[1] = (size_t)size_x;
			R[2] = (size_t)size_y;
			R[3] = (size_t)size_z;
			T *S = (T*)malloc(4 * sizeof(T));					// spacing in 4 dimension
			S[0] = 1.0f;
			S[1] = dx;
			S[2] = dy;
			S[3] = dz;
			size_t num = size_x * size_y;

			HANDLE_ERROR(cudaMalloc((void**)&d_R, 4 * sizeof(size_t)));
			HANDLE_ERROR(cudaMalloc((void**)&d_S, 4 * sizeof(T)));
			HANDLE_ERROR(cudaMemcpy(d_R, R, 4 * sizeof(size_t), cudaMemcpyHostToDevice));
			HANDLE_ERROR(cudaMemcpy(d_S, S, 4 * sizeof(T), cudaMemcpyHostToDevice));

			// for every slice of image
			unsigned p = 0;																// percentage of progress
			for (int i = 0; i < size_z; i++) {

				int x = 0 - (int)Xl;					// translate whole network(including inlet/outlet) to origin
				int y = 0 - (int)Yb;
				int z = i + (int)center[2];				// box symmetric along z-axis
				// allocate image slice memory
				unsigned char* d_ptr;
				unsigned char* ptr = (unsigned char*)malloc(num * sizeof(unsigned char));
				memset(ptr, 0, num * sizeof(unsigned char));

				HANDLE_ERROR(cudaMalloc((void**)&d_ptr, num * sizeof(unsigned char)));
				if (prototype)							// load prototype image stack if provided
					HANDLE_ERROR(cudaMemcpy(d_ptr, &I.data()[i * num], num * sizeof(unsigned char), cudaMemcpyHostToDevice));

				cudaDeviceProp prop;
				cudaGetDeviceProperties(&prop, 0);										// get cuda device properties structure
				size_t max_thread = (size_t)sqrt(prop.maxThreadsPerBlock);				// get the maximum number of thread per block

				dim3 block((unsigned)(size_x / max_thread + 1), (unsigned)(size_y / max_thread + 1));
				dim3 thread((unsigned)max_thread, (unsigned)max_thread);
				inside_sphere << <block, thread >> > (d_V, A.size(), Z, d_R, d_S, d_ptr, x, y, z);
				cudaDeviceSynchronize();
				inside_cone << <block, thread >> > (d_E, B.size(), Z, d_R, d_S, d_ptr, x, y, z);
				cudaDeviceSynchronize();
				inside_cuboid << <block, thread >> > (d_B, CU.size(), Z, d_R, d_S, d_ptr, x, y, z);

				HANDLE_ERROR(cudaMemcpy(ptr, d_ptr, num * sizeof(unsigned char), cudaMemcpyDeviceToHost));

				I.set(ptr, i);

				free(ptr);
				HANDLE_ERROR(cudaFree(d_ptr));

				// print progress bar
				p = (unsigned int)((i + 1) / size_z * 100);
				rtsProgressBar(p);
			}

			// clear up
			free(R);
			free(S);
			HANDLE_ERROR(cudaFree(d_R));
			HANDLE_ERROR(cudaFree(d_S));
			HANDLE_ERROR(cudaFree(d_V));
			HANDLE_ERROR(cudaFree(d_E));
			HANDLE_ERROR(cudaFree(d_B));

			if (stackdir == "")
				I.save_images("image????.bmp");
			else
				I.save_images(stackdir + "/image????.bmp");
		}

		/// save network flow profile
		void save_network() {
			
			// save the pressure information to CSV file
			std::string p_filename = "profile/pressure.csv";
			std::ofstream p_file;
			p_file.open(p_filename.c_str());
			p_file << "Vertex, Pressure(g/" << units << "/s^2)" << std::endl;
			for (unsigned i = 0; i < num_vertex; i++)
				p_file << i << "," << pressure[i] << std::endl;
			p_file.close();

			// save the flow information to CSV file
			std::string f_filename = "profile/flow_rate.csv";
			std::ofstream f_file;
			f_file.open(f_filename.c_str());
			f_file << "Edge, Volume flow rate(" << units << "^3/s)" << std::endl;
			for (unsigned i = 0; i < num_edge; i++)
				f_file << Q[i].first << "->" << Q[i].second << "," << Q[i].third << std::endl;
			f_file.close();
		}

		/// check whether current network needs to be subdivided
		bool check_subdivision() {
			
			for (size_t i = 0; i < num_edge; i++) {
				if (E[i].size() > 2) {
					return true;
				}
			}
			return false;
		}

		/// calculate the inverse of A and store the result in C
		void inversion(T** A, int order, T* C) {

#ifdef __CUDACC__

			// convert from double pointer to single pointer, make it flat
			T* Aflat = (T*)malloc(order * order * sizeof(T));
			for (int i = 0; i < order; i++)
				for (int j = 0; j < order; j++)
					Aflat[i * order + j] = A[i][j];

			// create device pointer
			T* d_Aflat;		// flat original matrix
			T* d_Cflat;	// flat inverse matrix
			T** d_A;		// put the flat original matrix into another array of pointer
			T** d_C;
			int *d_P;
			int *d_INFO;

			// allocate memory on device
			HANDLE_ERROR(cudaMalloc((void**)&d_Aflat, order * order * sizeof(T)));
			HANDLE_ERROR(cudaMalloc((void**)&d_Cflat, order * order * sizeof(T)));
			HANDLE_ERROR(cudaMalloc((void**)&d_A, sizeof(T*)));
			HANDLE_ERROR(cudaMalloc((void**)&d_C, sizeof(T*)));
			HANDLE_ERROR(cudaMalloc((void**)&d_P, order * 1 * sizeof(int)));
			HANDLE_ERROR(cudaMalloc((void**)&d_INFO, 1 * sizeof(int)));

			// copy matrix from host to device
			HANDLE_ERROR(cudaMemcpy(d_Aflat, Aflat, order * order * sizeof(T), cudaMemcpyHostToDevice));

			// copy matrix from device to device
			HANDLE_ERROR(cudaMemcpy(d_A, &d_Aflat, sizeof(T*), cudaMemcpyHostToDevice));
			HANDLE_ERROR(cudaMemcpy(d_C, &d_Cflat, sizeof(T*), cudaMemcpyHostToDevice));

			// calculate the inverse of matrix based on cuBLAS
			cublasHandle_t handle;
			CUBLAS_HANDLE_ERROR(cublasCreate_v2(&handle));	// create cuBLAS handle object

			CUBLAS_HANDLE_ERROR(cublasSgetrfBatched(handle, order, d_A, order, d_P, d_INFO, 1));

			int INFO = 0;
			HANDLE_ERROR(cudaMemcpy(&INFO, d_INFO, sizeof(int), cudaMemcpyDeviceToHost));
			if (INFO == order)
			{
				std::cout << "Factorization Failed : Matrix is singular." << std::endl;
				cudaDeviceReset();
				exit(1);
			}

			CUBLAS_HANDLE_ERROR(cublasSgetriBatched(handle, order, (const T **)d_A, order, d_P, d_C, order, d_INFO, 1));

			CUBLAS_HANDLE_ERROR(cublasDestroy_v2(handle));

			// copy inverse matrix from device to device
			HANDLE_ERROR(cudaMemcpy(&d_Cflat, d_C, sizeof(T*), cudaMemcpyDeviceToHost));

			// copy inverse matrix from device to host
			HANDLE_ERROR(cudaMemcpy(C, d_Cflat, order * order * sizeof(T), cudaMemcpyDeviceToHost));

			// clear up
			free(Aflat);
			HANDLE_ERROR(cudaFree(d_Aflat));
			HANDLE_ERROR(cudaFree(d_Cflat));
			HANDLE_ERROR(cudaFree(d_A));
			HANDLE_ERROR(cudaFree(d_C));
			HANDLE_ERROR(cudaFree(d_P));
			HANDLE_ERROR(cudaFree(d_INFO));

#else
			// get the determinant of a
			double det = 1.0 / determinant(A, order);

			// memory allocation
			T* tmp = (T*)malloc((order - 1)*(order - 1) * sizeof(T));
			T** minor = (T**)malloc((order - 1) * sizeof(T*));
			for (int i = 0; i < order - 1; i++)
				minor[i] = tmp + (i * (order - 1));

			for (int j = 0; j < order; j++) {
				for (int i = 0; i < order; i++) {
					// get the co-factor (matrix) of A(j,i)
					get_minor(A, minor, j, i, order);
					C[i][j] = det * determinant(minor, order - 1);
					if ((i + j) % 2 == 1)
						C[i][j] = -C[i][j];
				}
			}

			// release memory
			free(tmp);
			free(minor);
#endif
		}

		/// arithmetic
		// assignment
		flow<T> & operator= (flow<T> rhs) {
			E = rhs.E;
			V = rhs.V;

			return *this;
		}
	};
}

#endif