#include <stdlib.h>
#include <string>
#include <fstream>
#include <algorithm>

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

#include "flow.h"

// STIM include
#include <stim/visualization/gl_aaboundingbox.h>
#include <stim/parser/arguments.h>
#include <stim/visualization/camera.h>
#include <stim/visualization/colormap.h>
#include <stim/cuda/cudatools/error.h>
#include <stim/grids/image_stack.h>


//********************parameter setting********************
// overall parameters
std::string units;										// units used in this program
int vX, vY;
float dx, dy, dz;										// x, y and z image scaling(units/pixel)
std::string stackdir = "";								// directory where image stacks will be stored
stim::arglist args;										// create an instance of arglist
stim::gl_aaboundingbox<float> bb;						// axis-aligned bounding box object
stim::camera cam;										// camera object
unsigned num_edge;										// number of edges in the network
unsigned num_vertex;									// number of vertex in the network
std::vector<unsigned> pendant_vertex;					// list of pendant vertex index in GT
std::vector<std::string> menu_option = { "simulation", "build inlet/outlet", "manufacture", "adjustment" };
stim::flow<float> flow;									// flow object
stim::flow<float> backup;								// flow backup
float move_pace;										// camera moving parameter
float u;												// viscosity
float rou;												// density
float max_v;
float min_v;
int mods;												// special keyboard input
std::vector<unsigned char> color;						// velocity color map
std::vector<int> velocity_bar;							// velocity bar
float length = 40.0f;									// cuboid length
float scale = 1.0f;										// scale factor
bool image_stack = false;								// flag indicates an image stack been loaded
stim::image_stack<unsigned char, float> S;				// image stack
float binary_threshold = 128;							// threshold for binary transformation
float in = 0.0f;										// total input volume flow rate
float out = 0.0f;
float Rt = 0.0f;										// total resistance
float Qn = 0.0f;										// required input total volume flow rate
GLint dlist;											// simulation display list
bool undo = false;										// delete display list

// hard-coded parameters
float camera_factor = 1.2f;			// start point of the camera as a function of X and Y size
float orbit_factor = 0.01f;			// degrees per pixel used to orbit the camera
float zoom_factor = 10.0f;			// zooming factor
float border_factor = 20.0f;		// border
float radii_factor = 1.0f;			// radii changing factor
GLint subdivision = 20;				// slices and stacks
float default_radius = 5.0f;		// default radii of network vertex
float delta = 0.01f;				// small discrepancy
float eps = 20.0f;					// epsilon threshold
float max_pressure = 0.0f;			// maximum pressure that the channel can bear
float height_threshold = 100.0f;	// connection height constraint
float fragment_ratio = 0.0f;		// fragment ratio

// glut event parameters
int mouse_x;						// window x-coordinate
int mouse_y;						// window y-coordinate
int picked_x;						// picked window x-coordinate
int picked_y;						// picked window y-coordinate
bool LTbutton = false;				// true means down while false means up		

// simulation parameters
bool render_direction = false;		// flag indicates rendering flow direction for one edge
bool simulation = false;			// flag indicates simulation mode
bool color_bound = false;			// flag indicates velocity color map bound
bool to_select_pressure = false;	// flag indicates having selected a vertex to modify pressure
bool mark_index = true;				// flag indicates marking the index near the vertex
bool glyph_mode = false;			// flag indicates rendering glyph for flow velocity field
bool frame_mode = false;			// flag indicates rendering filament framing structrues
bool subdivided = false;			// flag indicates subdivision status
unsigned pressure_index;			// the index of vertex that is clicked
unsigned direction_index = UINT_MAX;// the index of edge that is pointed at
std::vector<stim::vec3<float> > back_vertex;	// vertex back up for marking indices

// build inlet/outlet parameters
bool build_inlet_outlet = false;	// flag indicates building inlets and outlets
bool modified_bridge = false;		// flag indicates having modified inlet/outlet connection
bool hilbert_curve = false;			// flag indicates enabling hilbert curves constructions
bool change_fragment = false;		// flag indicates changing fragment for square wave connections
bool picked_connection = false;		// flag indicates picked one connection
bool render_new_connection = false;	// flag indicates rendering new line connection in trasparency
bool redisplay = false;				// flag indicates redisplay rendering
bool connection_done = false;		// flag indicates finishing connections
bool render_flow_rate = false;		// flag indicates rendering total volume flow rate
unsigned connection_index = UINT_MAX;// the index of connection that is picked
unsigned port_index = 0;			// inlet (0) or outlet (1)
stim::vec3<float> tmp_v1, tmp_v2;	// temp vertex
int coef;							// computational coefficient factor

// manufacture parameters
bool manufacture = false;			// flag indicates manufacture mode


//********************helper function*********************
// get the network basic information
inline void get_background() {

	pendant_vertex = flow.get_pendant_vertex();
	num_edge = flow.edges();
	num_vertex = flow.vertices();

	// set the initial radii
	flow.init(num_edge, num_vertex);			// initialize flow object
	
	// if no radius information laoded
	if (!flow.get_radius(0, 0))
		for (unsigned i = 0; i < num_edge; i++)
			flow.set_r(i, default_radius);
}

// convert from window coordinates to world coordinates
inline void window_to_world(GLdouble &x, GLdouble &y, GLdouble &z) {

	GLint    viewport[4];
	GLdouble modelview[16];
	GLdouble projection[16];
	GLdouble winX, winY;
	GLfloat  winZ;

	glGetIntegerv(GL_VIEWPORT, viewport);
	glGetDoublev(GL_MODELVIEW_MATRIX, modelview);
	glGetDoublev(GL_PROJECTION_MATRIX, projection);

	winX = (GLdouble)mouse_x;
	winY = viewport[3] - (GLdouble)mouse_y;
	glReadPixels((GLint)winX, (GLint)winY, (GLsizei)1, (GLsizei)1, GL_DEPTH_COMPONENT, GL_FLOAT, &winZ);
	gluUnProject(winX, winY, winZ, modelview, projection, viewport, &x, &y, &z);
}

// convert current image stack into a binary mask
#ifdef __CUDACC__
template <typename T, typename F>
__global__ void binary_transform(size_t N, T* ptr, F threshold) {

	size_t ix = blockDim.x * blockIdx.x + threadIdx.x;
	if (ix >= N) return;					// avoid seg-fault

	if (ptr[ix] >= threshold)				// binary transformation
		ptr[ix] = 0;
	else
		ptr[ix] = 255;
}
#endif


//********************simulation function**********************
// initialize flow object
void flow_initialize() {

	flow.set = true;
	stim::vec3<float> center = bb.center();
	flow.P.clear();
	flow.P.resize(num_vertex, 0);									// clear up initialized pressure

	for (unsigned i = 0; i < pendant_vertex.size(); i++) {
		if (flow.get_vertex(pendant_vertex[i])[0] <= center[0])
			flow.P[pendant_vertex[i]] = max_pressure - i * delta;	// should set minor discrepancy
		else
			flow.P[pendant_vertex[i]] = (i + 1) * delta;			// algorithm treat 0 as no initial pressure
	}
}

// find the stable flow state
void flow_stable_state() {
	
	flow.solve_flow(u);
	flow.get_color_map(max_v, min_v, color, pendant_vertex);
	color_bound = true;

	velocity_bar.resize(num_edge);
	for (unsigned i = 0; i < num_edge; i++)
		velocity_bar[i] = i;
	std::sort(velocity_bar.begin(), velocity_bar.end(), [&](int x, int y) {return abs(flow.v[x]) < abs(flow.v[y]); });
}

// adjustment on input volume flow rate and corresponding flow simulation
void adjustment() {

	system("CLS");							// clear up console box
	std::cout << "Please enter the input total volume flow rate: " << std::endl;
	std::cin >> Qn;

	flow.adjust(in, out, Rt, Qn, u);
}


//********************glut function********************
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

// set up the squash transform to whole screen
void glut_projection() {

	glMatrixMode(GL_PROJECTION);					// load the projection matrix for editing
	glLoadIdentity();								// start with the identity matrix
	vX = glutGet(GLUT_WINDOW_WIDTH);				// use the whole screen for rendering
	vY = glutGet(GLUT_WINDOW_HEIGHT);
	glViewport(0, 0, vX, vY);						// specify a viewport for the entire window
	float aspect = (float)vX / (float)vY;			// calculate the aspect ratio
	gluPerspective(60, aspect, 0.1, 1000000);		// set up a perspective projection
}

// translate camera to origin
void glut_modelview() {

	glMatrixMode(GL_MODELVIEW);						// load the modelview matrix for editing
	glLoadIdentity();								// start with the identity matrix
	stim::vec3<float> eye = cam.getPosition();		// get the camera position (eye point)
	stim::vec3<float> focus = cam.getLookAt();		// get the camera focal point
	stim::vec3<float> up = cam.getUp();				// get the camera "up" orientation

	gluLookAt(eye[0], eye[1], eye[2], focus[0], focus[1], focus[2], up[0], up[1], up[2]);	// set up the OpenGL camera
}

// glut render function
void glut_render() {

	glEnable(GL_DEPTH_TEST);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
	glut_projection();
	glut_modelview();

	if (!simulation && !build_inlet_outlet || manufacture) {
		glColor3f(0.0f, 0.0f, 0.0f);
		flow.glCylinder0(scale, undo);
	}
	else {	
		flow.bounding_box();		// bounding box
		if (num_vertex > 100) {						// if the network is big enough (say 100), use display list
			if (undo) {								// undo rendering list
				undo = false;
				glDeleteLists(dlist, 1);
			}							
			if (!glIsList(dlist)) {
				dlist = glGenLists(1);
				glNewList(dlist, GL_COMPILE);
				// render network
				if (!glyph_mode) {
					flow.glSolidSphere(max_pressure, subdivision, scale);
					if (mark_index)
						flow.mark_vertex(back_vertex, scale);
					//flow.glSolidCone(subdivision);
					flow.glSolidCylinder(direction_index, color, subdivision, scale);
				}
				// render glyphs
				else
					flow.glyph(color, subdivision, scale, frame_mode);
				
				glEndList();
			}
			glCallList(dlist);
		}
		else {										// small network
			// render network
			if (!glyph_mode) {
				flow.glSolidSphere(max_pressure, subdivision, scale);
				if (mark_index) {
					flow.mark_vertex(back_vertex, scale);
					//flow.mark_edge();
				}
				//flow.glSolidCone(subdivision);
				flow.glSolidCylinder(direction_index, color, subdivision, scale);
			}
			// render glyphs
			else
				flow.glyph(color, subdivision, scale, frame_mode);
		}
		flow.glSolidCuboid(subdivision, manufacture, length);		// render bus source
		if (render_direction && !glyph_mode)						// render the flow direction of the vertex pointed
			flow.glSolidCone(direction_index, subdivision, scale);
	}

	if (build_inlet_outlet)
		flow.line_bridge(redisplay);

	if (manufacture) {
		flow.glSolidCuboid(subdivision, manufacture, length);
		flow.tube_bridge(redisplay, subdivision, scale);
	}

	if (picked_connection && render_new_connection) {
		glEnable(GL_BLEND);
		glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
		glColor4f(0.0f, 0.0f, 0.0f, 0.4f);
		glBegin(GL_LINE_STRIP);
		if (!port_index) {
			glVertex3f(flow.inlet[connection_index].V[1][0], flow.inlet[connection_index].V[1][1], flow.inlet[connection_index].V[1][2]);
			glVertex3f(tmp_v1[0], tmp_v1[1], tmp_v1[2]);
			glVertex3f(tmp_v2[0], tmp_v2[1], tmp_v2[2]);
			glVertex3f(flow.inlet[connection_index].V[2][0], flow.inlet[connection_index].V[2][1], flow.inlet[connection_index].V[2][2]);
		}
		else {
			glVertex3f(flow.outlet[connection_index].V[1][0], flow.outlet[connection_index].V[1][1], flow.outlet[connection_index].V[1][2]);
			glVertex3f(tmp_v1[0], tmp_v1[1], tmp_v1[2]);
			glVertex3f(tmp_v2[0], tmp_v2[1], tmp_v2[2]);
			glVertex3f(flow.outlet[connection_index].V[2][0], flow.outlet[connection_index].V[2][1], flow.outlet[connection_index].V[2][2]);
		}
		glEnd();
		glFlush();
		glDisable(GL_BLEND);
	}
	
	// render bars
	// bring up a pressure bar on left
	if (to_select_pressure) {
		
		glMatrixMode(GL_PROJECTION);									// set up the 2d viewport for mode text printing
		glPushMatrix();
		glLoadIdentity();
		vX = glutGet(GLUT_WINDOW_WIDTH);								// get the current window width
		vY = glutGet(GLUT_WINDOW_HEIGHT);								// get the current window height
		glViewport(0, 0, vX, vY);										// locate to left bottom corner
		gluOrtho2D(0, vX, 0, vY);										// define othogonal aspect

		glMatrixMode(GL_MODELVIEW);
		glPushMatrix();
		glLoadIdentity();

		glLineWidth(border_factor);
		glBegin(GL_LINES);
		glColor3f(0.0, 0.0, 1.0);										// blue to red
		glVertex2f(border_factor, border_factor);
		glColor3f(1.0, 0.0, 0.0);
		glVertex2f(border_factor, (vY - 2.0f * border_factor));
		glEnd();
		glFlush();

		// pressure bar text
		glColor3f(0.0f, 0.0f, 0.0f);
		glRasterPos2f(0.0f, vY - border_factor);
		std::stringstream ss_p;
		ss_p << "Pressure Bar";
		glutBitmapString(GLUT_BITMAP_HELVETICA_18, (const unsigned char*)(ss_p.str().c_str()));

		// pressure range text
		float step = vY - 3.0f * border_factor;
		step /= 10;
		for (unsigned i = 0; i < 11; i++) {
			glRasterPos2f((border_factor * 1.5f), (border_factor + i * step));
			std::stringstream ss_n;
			ss_n << (float)i * max_pressure / 10;
			glutBitmapString(GLUT_BITMAP_HELVETICA_18, (const unsigned char*)(ss_n.str().c_str()));
		}
		glPopMatrix();
		glMatrixMode(GL_PROJECTION);
		glPopMatrix();
	}

	// bring up a velocity bar on left
	if ((simulation || build_inlet_outlet) && !to_select_pressure && !change_fragment) {
		
		glMatrixMode(GL_PROJECTION);									// set up the 2d viewport for mode text printing
		glPushMatrix();
		glLoadIdentity();
		vX = glutGet(GLUT_WINDOW_WIDTH);								// get the current window width
		vY = glutGet(GLUT_WINDOW_HEIGHT);								// get the current window height
		glViewport(0, 0, vX, vY);										// locate to left bottom corner
		gluOrtho2D(0, vX, 0, vY);										// define othogonal aspect

		glMatrixMode(GL_MODELVIEW);
		glPushMatrix();
		glLoadIdentity();

		float step = (vY - 3 * border_factor);
		step /= BREWER_CTRL_PTS - 1;
		for (unsigned i = 0; i < BREWER_CTRL_PTS - 1; i++) {
			glLineWidth(border_factor);
			glBegin(GL_LINES);
			glColor3f(BREWERCP[i * 4 + 0], BREWERCP[i * 4 + 1], BREWERCP[i * 4 + 2]);
			glVertex2f(border_factor, border_factor + i * step);
			glColor3f(BREWERCP[(i + 1) * 4 + 0], BREWERCP[(i + 1) * 4 + 1], BREWERCP[(i + 1) * 4 + 2]);
			glVertex2f(border_factor, border_factor + (i + 1) * step);
			glEnd();
		}
		glFlush();

		// pressure bar text
		glColor3f(0.0f, 0.0f, 0.0f);
		glRasterPos2f(0.0f, vY - border_factor);
		std::stringstream ss_p;
		ss_p << "Velocity range";
		glutBitmapString(GLUT_BITMAP_HELVETICA_18, (const unsigned char*)(ss_p.str().c_str()));

		// pressure range text
		step = vY - 3 * border_factor;
		step /= 10;
		for (unsigned i = 0; i < 11; i++) {
			glRasterPos2f(border_factor * 1.5f, border_factor + i * step);
			std::stringstream ss_n;
			ss_n << min_v + i * (max_v - min_v) / 10;
			glutBitmapString(GLUT_BITMAP_HELVETICA_18, (const unsigned char*)(ss_n.str().c_str()));
		}
		glPopMatrix();
		glMatrixMode(GL_PROJECTION);
		glPopMatrix();
	}

	// bring up a ratio bar on the left
	if (change_fragment) {
		
		glMatrixMode(GL_PROJECTION);									// set up the 2d viewport for mode text printing
		glPushMatrix();
		glLoadIdentity();
		vX = glutGet(GLUT_WINDOW_WIDTH);								// get the current window width
		vY = glutGet(GLUT_WINDOW_HEIGHT);								// get the current window height
		glViewport(0, 0, vX, vY);										// locate to left bottom corner
		gluOrtho2D(0, vX, 0, vY);										// define othogonal aspect

		glMatrixMode(GL_MODELVIEW);
		glPushMatrix();
		glLoadIdentity();

		glLineWidth(border_factor);
		glBegin(GL_LINES);
		glColor3f(0.0, 0.0, 1.0);										// blue to red
		glVertex2f(border_factor, border_factor);
		glColor3f(1.0, 0.0, 0.0);
		glVertex2f(border_factor, (vY - 2.0f * border_factor));
		glEnd();
		glFlush();

		// ratio bar text
		glColor3f(0.0f, 0.0f, 0.0f);
		glRasterPos2f(0.0f, vY - border_factor);
		std::stringstream ss_p;
		ss_p << "Ratio bar";
		glutBitmapString(GLUT_BITMAP_HELVETICA_18, (const unsigned char*)(ss_p.str().c_str()));

		// ratio range text
		float step = vY - 3.0f * border_factor;
		step /= 10;
		for (unsigned i = 0; i < 11; i++) {
			glRasterPos2f((border_factor * 1.5f), (border_factor + i * step));
			std::stringstream ss_n;
			ss_n << (float)i * 1.0f / 10;
			glutBitmapString(GLUT_BITMAP_HELVETICA_18, (const unsigned char*)(ss_n.str().c_str()));
		}
		glPopMatrix();
		glMatrixMode(GL_PROJECTION);
		glPopMatrix();
	}

	if (build_inlet_outlet)
		if (render_flow_rate)
			flow.display_flow_rate(in, out);

	glutSwapBuffers();
}

// register glut menu options
void glut_menu(int value) {

	int num = glutGet(GLUT_MENU_NUM_ITEMS);
	if (value == 1) {
		simulation = true;
		build_inlet_outlet = false;
		render_flow_rate = false;
		manufacture = false;
		modified_bridge = false;
		change_fragment = false;
		connection_done = false;
		// first time
		if (!flow.set) {						// only first time simulation called "simulation", ^_^
			get_background();					// get the graph information
			back_vertex = flow.back_vertex();	// vertex back up for marking indices
			flow_initialize();					// initialize flow condition
			menu_option[0] = "resimulation";
		}

		// simulation / resimulation
		flow_stable_state();					// main function of solving the linear system
		flow.print_flow();
		
		if (!glyph_mode)
			glut_set_menu(num, 2);
	}

	if (value == 2) {
		simulation = false;
		build_inlet_outlet = true;
		manufacture = false;
		if (!modified_bridge && !connection_done) {
			flow.set_main_feeder();
			flow.build_synthetic_connection(u, default_radius);
			flow.check_direct_connection();	// check whether direct connections intersect each other
			connection_done = true;
		}
		else if (modified_bridge) {
			modified_bridge = false;
			redisplay = true;
			flow.clear_synthetic_connection();
		}

		glut_set_menu(num, 4);
	}

	if (value == 3) {
		simulation = false;
		build_inlet_outlet = false;
		manufacture = true;
		glyph_mode = false;			// manufacuture mode doesn't need flow direction
		redisplay = true;
	}

	if (value == 4) {
		simulation = true;
		build_inlet_outlet = false;
		render_flow_rate = false;
		manufacture = false;
		
		adjustment();					// adjust network flow accordingly

		glut_set_menu(num, 1);
	}

	glutPostRedisplay();
}

// defines camera motion based on mouse dragging
void glut_motion(int x, int y) {

	mods = glutGetModifiers();
	if (LTbutton && mods == 0) {

		float theta = orbit_factor * (mouse_x - x);		// determine the number of degrees along the x-axis to rotate
		float phi = orbit_factor * (y - mouse_y);		// number of degrees along the y-axis to rotate

		cam.OrbitFocus(theta, phi);						// rotate the camera around the focal point
	}
	mouse_x = x;										// update the mouse position
	mouse_y = y;

	glutPostRedisplay();								// re-draw the visualization
}

// defines passive mouse motion function
void glut_passive_motion(int x, int y) {

	mods = glutGetModifiers();

	// check whether the mouse point near to an edge
	GLdouble posX, posY, posZ;
	window_to_world(posX, posY, posZ);			// get the world coordinates

	if (simulation || build_inlet_outlet && !mods) {
		bool flag = flow.epsilon_edge((float)posX, (float)posY, (float)posZ, eps, direction_index);
		if (flag && !glyph_mode)
			render_direction = true;
		else if (!flag && !glyph_mode) {
			if (render_direction)				// if the direction is displaying currently, do a short delay
				Sleep(300);
			render_direction = false;
			direction_index = -1;
		}
		undo = true;
	}

	if (mods == GLUT_ACTIVE_SHIFT && picked_connection) {
		render_new_connection = true;
		size_t i;
		if (!port_index) {
			tmp_v1 = stim::vec3<float>(flow.inlet[connection_index].V[1][0], flow.inlet[connection_index].V[1][1] + (float)(picked_y - y), flow.inlet[connection_index].V[1][2]);
			tmp_v2 = stim::vec3<float>(flow.inlet[connection_index].V[2][0], flow.inlet[connection_index].V[2][1] + (float)(picked_y - y), flow.inlet[connection_index].V[2][2]);
			i = flow.inlet[connection_index].V.size();
			if (coef * tmp_v1[1] < coef * flow.inlet[connection_index].V[i - 1][1]) {
				tmp_v1[1] = flow.inlet[connection_index].V[i - 1][1];
				tmp_v2[1] = flow.inlet[connection_index].V[i - 1][1];
			}
		}
		else {
			tmp_v1 = stim::vec3<float>(flow.outlet[connection_index].V[1][0], flow.outlet[connection_index].V[1][1] + (float)(picked_y - y), flow.outlet[connection_index].V[1][2]);
			tmp_v2 = stim::vec3<float>(flow.outlet[connection_index].V[2][0], flow.outlet[connection_index].V[2][1] + (float)(picked_y - y), flow.outlet[connection_index].V[2][2]);
			i = flow.outlet[connection_index].V.size();
			if (coef * tmp_v1[1] < coef * flow.outlet[connection_index].V[i - 1][1]) {
				tmp_v1[1] = flow.outlet[connection_index].V[i - 1][1];
				tmp_v2[1] = flow.outlet[connection_index].V[i - 1][1];
			}
		}	
	}
	else if (mods == GLUT_ACTIVE_CTRL && picked_connection) {
		render_new_connection = true;
		if (!port_index) {
			tmp_v1 = stim::vec3<float>(flow.inlet[connection_index].V[0][0] + (float)(x - picked_x), flow.inlet[connection_index].V[0][1], flow.inlet[connection_index].V[0][2]);
			tmp_v2 = stim::vec3<float>(flow.inlet[connection_index].V[1][0] + (float)(x - picked_x), flow.inlet[connection_index].V[1][1], flow.inlet[connection_index].V[1][2]);
			if (tmp_v1[0] < flow.main_feeder[port_index][0] - length / 2) {
				tmp_v1[0] = flow.main_feeder[port_index][0] - length / 2;
				tmp_v2[0] = flow.main_feeder[port_index][0] - length / 2;
			}
			else if (tmp_v1[0] > flow.main_feeder[port_index][0] + length / 2) {
				tmp_v1[0] = flow.main_feeder[port_index][0] + length / 2;
				tmp_v2[0] = flow.main_feeder[port_index][0] + length / 2;
			}
		}
		else {
			tmp_v1 = stim::vec3<float>(flow.outlet[connection_index].V[0][0] + (float)(x - picked_x), flow.outlet[connection_index].V[0][1], flow.outlet[connection_index].V[0][2]);
			tmp_v2 = stim::vec3<float>(flow.outlet[connection_index].V[1][0] + (float)(x - picked_x), flow.outlet[connection_index].V[1][1], flow.outlet[connection_index].V[1][2]);
			if (tmp_v1[0] > flow.main_feeder[port_index][0] + length / 2) {
				tmp_v1[0] = flow.main_feeder[port_index][0] + length / 2;
				tmp_v2[0] = flow.main_feeder[port_index][0] + length / 2;
			}
			else if (tmp_v1[0] < flow.main_feeder[port_index][0] - length / 2) {
				tmp_v1[0] = flow.main_feeder[port_index][0] - length / 2;
				tmp_v2[0] = flow.main_feeder[port_index][0] - length / 2;
			}
		}
	}
	else
		render_new_connection = false;

	mouse_x = x;
	mouse_y = y;

	glutPostRedisplay();							// re-draw the visualization
}

// get click window coordinates
void glut_mouse(int button, int state, int x, int y) {

	mods = glutGetModifiers();						// get special keyboard input

	mouse_x = x;
	mouse_y = y;
	if (!mods) {
		picked_connection = false;
		render_new_connection = false;
	}
	if (button == GLUT_LEFT_BUTTON && state == GLUT_DOWN)
		LTbutton = true;
	else if (button == GLUT_LEFT_BUTTON && state == GLUT_UP)
		LTbutton = false;

	if (button == GLUT_LEFT_BUTTON && state == GLUT_DOWN && !mods && simulation && !to_select_pressure) {
		GLdouble posX, posY, posZ;
		window_to_world(posX, posY, posZ);			// get the world coordinates

		bool flag = flow.epsilon_vertex((float)posX, (float)posY, (float)posZ, eps, scale, pressure_index);
		if (flag) {
			std::vector<unsigned>::iterator it = std::find(pendant_vertex.begin(), pendant_vertex.end(), pressure_index);
			if (it != pendant_vertex.end()) 		// if it is dangle vertex
				to_select_pressure = true;
		}
	}
	else if (button == GLUT_LEFT_BUTTON && state == GLUT_DOWN && !mods && simulation && to_select_pressure) {
		if (y >= 2 * border_factor && y <= vY - border_factor) {	// within the pressure bar range
			to_select_pressure = false;
			float tmp_pressure = (float)(vY - y - border_factor) / ((float)vY - 3.0f * border_factor) * max_pressure;
			flow.set_pressure(pressure_index, tmp_pressure);
			//flow_stable_state();									// main function of solving the linear system
			//flow.print_flow();
		}
	}
	else if (button == GLUT_LEFT_BUTTON && state == GLUT_DOWN && !mods && modified_bridge && change_fragment) {
		if (y >= 2 * border_factor && y <= vY - border_factor) 		// within the ratio bar range
			fragment_ratio = (float)(vY - y - border_factor) / ((float)vY - 3.0f * border_factor) * 1.0f;
		else if (y < 2 * border_factor)
			fragment_ratio = 1.0f;
		else if (y > vY - border_factor)
			fragment_ratio = 0.0f;

		change_fragment = false;
		render_flow_rate = true;
		flow.modify_synthetic_connection(u, rou, hilbert_curve, height_threshold, in, out, 2, fragment_ratio, default_radius);
	}
	// move connections along y-axis
	else if (button == GLUT_LEFT_BUTTON && state == GLUT_DOWN && mods == GLUT_ACTIVE_SHIFT && !modified_bridge && !picked_connection) {	
		GLdouble posX, posY, posZ;
		window_to_world(posX, posY, posZ);			// get the world coordinates

		bool flag = flow.epsilon_edge((float)posX, (float)posY, (float)posZ, eps, connection_index, port_index);
		if (flag) {
			picked_connection = true;
			picked_x = x;
			picked_y = y;
			if (!port_index)
				if (flow.inlet[connection_index].V[2][1] > flow.main_feeder[port_index][1])
					coef = 1;
				else
					coef = -1;
			else
				if (flow.outlet[connection_index].V[2][1] > flow.main_feeder[port_index][1])
					coef = 1;
				else
					coef = -1;
		}
		else
			picked_connection = false;
	}
	else if (button == GLUT_LEFT_BUTTON && state == GLUT_DOWN && mods == GLUT_ACTIVE_SHIFT && !modified_bridge && render_new_connection) {
		float l = 0.0f;
		std::vector<typename stim::vec3<float> > V;
		size_t i;
		if (!port_index) {
			i = flow.inlet[connection_index].V.size();
			if (tmp_v2[1] != flow.inlet[connection_index].V[i - 1][1]) {
				V.resize(4);
				V[0] = flow.inlet[connection_index].V[0];
				V[1] = tmp_v1;
				V[2] = tmp_v2;
				V[3] = flow.inlet[connection_index].V[i - 1];
				std::swap(flow.inlet[connection_index].V, V);
			}
			else {
				V.resize(3);
				V[0] = flow.inlet[connection_index].V[0];
				V[1] = tmp_v1;
				V[2] = tmp_v2;
				std::swap(flow.inlet[connection_index].V, V);
			}
			// calculate new length
			for (unsigned i = 0; i < flow.inlet[connection_index].V.size() - 1; i++) {
				l += (flow.inlet[connection_index].V[i + 1] - flow.inlet[connection_index].V[i]).len();
			}
			flow.inlet[connection_index].l = l;
		}
		else {
			i = flow.outlet[connection_index].V.size();
			if (tmp_v2[1] != flow.outlet[connection_index].V[i - 1][1]) {
				V.resize(4);
				V[0] = flow.outlet[connection_index].V[0];
				V[1] = tmp_v1;
				V[2] = tmp_v2;
				V[3] = flow.outlet[connection_index].V[i - 1];
				std::swap(flow.outlet[connection_index].V, V);
			}
			else {
				V.resize(3);
				V[0] = flow.outlet[connection_index].V[0];
				V[1] = tmp_v1;
				V[2] = tmp_v2;
				std::swap(flow.outlet[connection_index].V, V);
			}
			// calculate new length
			for (unsigned i = 0; i < flow.outlet[connection_index].V.size() - 1; i++) {
				l += (flow.outlet[connection_index].V[i + 1] - flow.outlet[connection_index].V[i]).len();
			}
			flow.outlet[connection_index].l = l;
		}

		redisplay = true;
		render_new_connection = false;
		picked_connection = false;

		flow.check_direct_connection();
		flow.backup();								// back up direct synthetic connections
	}
	// move connections along x-axis
	else if (button == GLUT_LEFT_BUTTON && state == GLUT_DOWN && mods == GLUT_ACTIVE_CTRL && !modified_bridge && !picked_connection) {
		GLdouble posX, posY, posZ;
		window_to_world(posX, posY, posZ);			// get the world coordinates

		bool flag = flow.epsilon_edge((float)posX, (float)posY, (float)posZ, eps, connection_index, port_index);
		if (flag) {
			picked_connection = true;
			picked_x = x;
			picked_y = y;
			if (!port_index)
				coef = 1;
			else
				coef = -1;
		}
		else
			picked_connection = false;
	}
	else if (button == GLUT_LEFT_BUTTON && state == GLUT_DOWN && mods == GLUT_ACTIVE_CTRL && !modified_bridge && render_new_connection) {
		float l = 0.0f;
		if (!port_index) {
			flow.inlet[connection_index].V[0] = tmp_v1;
			flow.inlet[connection_index].V[1] = tmp_v2;
			// calculate new length
			for (unsigned i = 0; i < flow.inlet[connection_index].V.size() - 1; i++) {
				l += (flow.inlet[connection_index].V[i + 1] - flow.inlet[connection_index].V[i]).len();
			}
			flow.inlet[connection_index].l = l;
		}
		else {
			flow.outlet[connection_index].V[0] = tmp_v1;
			flow.outlet[connection_index].V[1] = tmp_v2;
			// calculate new length
			for (unsigned i = 0; i < flow.outlet[connection_index].V.size() - 1; i++) {
				l += (flow.outlet[connection_index].V[i + 1] - flow.outlet[connection_index].V[i]).len();
			}
			flow.outlet[connection_index].l = l;
		}

		redisplay = true;
		render_new_connection = false;
		picked_connection = false;

		flow.check_direct_connection();
		flow.backup();
	}
}

// define camera move based on mouse wheel move
void glut_wheel(int wheel, int direction, int x, int y) {
	
	mods = glutGetModifiers();

	mouse_x = x;
	mouse_y = y;

	GLdouble posX, posY, posZ;
	window_to_world(posX, posY, posZ);			// get the world coordinates

	if (!to_select_pressure && (simulation || build_inlet_outlet || manufacture)) {							// check current pixel position only in simualtion and build_inlet_outlet modes
		bool flag = flow.epsilon_vertex((float)posX, (float)posY, (float)posZ, eps, scale, pressure_index);
		if (flag && simulation && !glyph_mode) {
			float tmp_r;
			if (direction > 0) {				// increase radii
				tmp_r = flow.get_radius(pressure_index);
				tmp_r += radii_factor;
			}
			else {
				tmp_r = flow.get_radius(pressure_index);
				tmp_r -= radii_factor;
				if (tmp_r <= 0)
					tmp_r = default_radius;
			}
			flow.set_radius(pressure_index, tmp_r);
			undo = true;									// undo rendering
		}
		else if (!mods) {
			if (direction > 0)								// if it is button 3(up), move closer
				move_pace = zoom_factor;
			else											// if it is button 4(down), leave farther
				move_pace = -zoom_factor;

			cam.Push(move_pace);
		}
	}

	// rescale
	if (mods == GLUT_ACTIVE_CTRL) {
		if (direction > 0) {
			if (scale >= 1)
				scale += 1.0f;
			else
				scale += 0.1f;
		}
		else {
			if (scale > 1)
				scale -= 1.0f;
			else if (scale <= 1 && scale > 0.1f)
				scale -= 0.1f;
			else
				scale = 1.0f;
		}
		undo = true;
		redisplay = true;
	}
	
	glutPostRedisplay();
}

// define keyboard inputs
void glut_keyboard(unsigned char key, int x, int y) {

	// register different keyboard operation
	switch (key) {

		// saving network flow profile
	case 's':
		flow.save_network();
		break;

		// convert network to binary format (.nwt)
	case 'c': {
		std::vector<std::string> tmp = stim::parser::split(args.arg(0), '.');
		std::stringstream ss;
		ss << tmp[0] << ".nwt";
		std::string filename = ss.str();
		flow.saveNwt(filename);
		break;
	}

		// subdivide current network for more detailed calculation
	case 'd': {
		// subdivide current network due to the limitation of current computation if needed
		if (!subdivided && simulation && !glyph_mode) {
			subdivided = true;

			// check whether current network can be subdivided
			if (flow.check_subdivision()) {
				flow.subdivision();
				get_background();
			}

			flow_initialize();					// initialize flow condition
			// resimulation
			flow_stable_state();				// main function of solving the linear system
			flow.print_flow();
			undo = true;
		}
		else if (subdivided && simulation && !glyph_mode) {
			subdivided = false;
			flow = backup;						// load back up

			get_background();
			flow_initialize();
			// resimulation
			flow_stable_state();				// main function of solving the linear system
			flow.print_flow();
			undo = true;
		}
		break;
	}

		// flow vector field visualization, Glyphs
	case 'f':
		if (glyph_mode && !manufacture && (simulation || build_inlet_outlet)) {
			glyph_mode = false;
			frame_mode = false;
			redisplay = true;							// lines and arrows rendering use the same display list
			int num = glutGet(GLUT_MENU_NUM_ITEMS);
			if (num == 1)
				glut_set_menu(num, 2);
		}
		else if (!glyph_mode && !manufacture && (simulation || build_inlet_outlet)) {
			glyph_mode = true;
			redisplay = true;
			int num = glutGet(GLUT_MENU_NUM_ITEMS);
			if (num == 2)
				glut_set_menu(num, 1);
		}
		undo = true;
		break;

		// filaments around arrows
	case 'g':
		if (glyph_mode) {
			if (frame_mode)
				frame_mode = false;
			else
				frame_mode = true;
		}
		undo = true;
		break;

		// open/close index marks
	case 'e':
		if (mark_index)
			mark_index = false;
		else
			mark_index = true;
		undo = true;
		break;

		// output image stack
	case 'm':
		if (manufacture) {
#ifdef __CUDACC__
			flow.make_image_stack(S, dx, dy, dz, stackdir, image_stack, default_radius, scale);

#else
			std::cout << "You need to have a gpu to make image stack, sorry." << std::endl;
#endif
		}
		else if (build_inlet_outlet && !modified_bridge) {
			modified_bridge = true;

			if (hilbert_curve)
				flow.modify_synthetic_connection(u, rou, hilbert_curve, height_threshold, in, out, 2, fragment_ratio);
			else
				change_fragment = true;
		}
		break;
	}
	
	glutPostRedisplay();
}

// glut initialization
void glut_initialize() {

	int myargc = 1;
	char* myargv[1];
	myargv[0] = strdup("generate_network_network");

	glutInit(&myargc, myargv);
	glutInitDisplayMode(GLUT_DEPTH | GLUT_DOUBLE | GLUT_RGBA);
	glutInitWindowPosition(100, 100);							// set the initial window position
	glutInitWindowSize(1000, 1000);
	glutCreateWindow("3D flow simulation");

	glutDisplayFunc(glut_render);
	glutMouseFunc(glut_mouse);
	glutMotionFunc(glut_motion);
	glutPassiveMotionFunc(glut_passive_motion);
	glutMouseWheelFunc(glut_wheel);
	glutKeyboardFunc(glut_keyboard);

	glutCreateMenu(glut_menu);					// create a menu object 
	glut_set_menu(0, 1);
	glutAttachMenu(GLUT_RIGHT_BUTTON);			// register right mouse to open menu option

	stim::vec3<float> c = bb.center();			// get the center of the network bounding box
	// place the camera along the z-axis at a distance determined by the network size along x and y
	cam.setPosition(c + stim::vec<float>(0, 0, camera_factor * std::max(bb.size()[0], bb.size()[1])));
	cam.LookAt(c[0], c[1], c[2]);
}

// output an advertisement for the lab, authors and usage information
void advertise() {
	std::cout << std::endl << std::endl;
	std::cout << " =======================================================================================" << std::endl;
	std::cout << "|Thank you for using the AFAN tool!                                                     |" << std::endl;
	std::cout << "|Scalable Tissue Imaging and Modeling (STIM) Lab, University of Houston                 |" << std::endl;
	std::cout << "|Developers: Jiaming Guo, David Mayerich                                                |" << std::endl;
	std::cout << "|Source: https://git.stim.ee.uh.edu/Jack/flow3.git									  |" << std::endl;
	std::cout << " =======================================================================================" << std::endl << std::endl;

	std::cout << "Usage(keyboard): e -> open/close indexing" << std::endl;
	std::cout << "                 m -> build synthetic connections(connection mode)/output augmented network as image stack (manufacture mode)" << std::endl;
	std::cout << "                 s -> save network flow profiles in profile folder as cvs files" << std::endl;
	std::cout << "                 c -> convert .obj network to .nwt network and stores in main folder" << std::endl;
	std::cout << "                 f -> open/close vector field visualization mode" << std::endl;
	std::cout << "                 g -> render filament frames in vector fiedl visualization mode" << std::endl;

	std::cout << args.str();
}

// main function: parse arguments and initialize GLUT
int main(int argc, char* argv[]) {
	
	// add arguments
	args.add("help", "prints the help");
	args.add("units", "string indicating units of length for output measurements (ex. velocity)", "um", "text string");
	args.add("maxpress", "maximum allowed pressure in g / units / s^2, default 2 is for blood when units = um", "2", "real value > 0");
	args.add("viscosity", "set the viscosity of the fluid (in g / units / s), default .00001 is for blood when units = um", ".00001", "real value > 0");
	args.add("rou", "set the desity of the fluid (in g / units^3), default 1.06*10^-12 is for blood when units = um", ".00000000000106", "real value > 0");
	args.add("hilbert", "activate hilbert curves connections");
	args.add("stack", "load the image stack");
	args.add("stackres", "spacing between pixel samples in each dimension(in units/pixel)", "1 1 1", "real value > 0");
	args.add("stackdir", "set the directory of the output image stack", "", "any existing directory (ex. /home/name/network)");
	args.add("scale", "scale down rendering fibers");
	args.add("lcc", "extract the largest connected component");
	
	args.parse(argc, argv);								// parse the command line

	if (args["help"].is_set()) {
		advertise();
		std::exit(1);
	}

	// load network
	if (args.nargs() == 0) {
		std::cout << "Network file required." << std::endl;
		return 1;
	}	
	else {						// load network from user 
		std::vector<std::string> tmp = stim::parser::split(args.arg(0), '.');
		if ("obj" == tmp[1]) {
			flow.load_obj(args.arg(0));
			backup.load_obj(args.arg(0));
		}	
		else if ("nwt" == tmp[1]) {		// stim network binary format
			flow.loadNwt(args.arg(0)); 
			backup.loadNwt(args.arg(0));
		}		
		else if ("swc" == tmp[1]) {
			flow.load_swc(args.arg(0)); 
			backup.load_swc(args.arg(0));
		}
		else {
			std::cout << "Invalid file type" << std::endl;
			std::exit(1);
		}
	}

	// extract the largest connected component

	// get the units to work on
	units = args["units"].as_string();
	flow.set_units(units);

	// blood pressure in capillaries range from 15 - 35 torr
	// 1 torr = 133.3 Pa
	max_pressure = (float)args["maxpress"].as_float();

	// normal blood viscosity range from 4 - 15 mPa·s(cP)
	// 1 Pa·s = 1 g / mm / s
	u = (float)args["viscosity"].as_float();			// g / units / s

	// normally the blood density in capillaries: 1060 kg/m^3 = 1.06*10^-12 g/um^3
	rou = (float)args["rou"].as_float();

	// check whether to enable hilbert curves or not
	hilbert_curve = args["hilbert"].is_set();

	// load image stack if provided
	if (args["stack"].is_set()) {
		image_stack = true;
		S.load_images(args["stack"].as_string());
		// binary transformation
#ifdef __CUDACC__
		size_t N = S.samples();													// number of pixels loaded
		unsigned char* d_S;														// image stack stored in device
		unsigned char* h_S = (unsigned char*)malloc(N * sizeof(unsigned char));	// image stack stored in host
		cudaMalloc((void**)&d_S, N * sizeof(unsigned char));
		cudaMemcpy(d_S, S.data(), N * sizeof(unsigned char), cudaMemcpyHostToDevice);

		size_t thread = 1024;
		size_t block = N / thread + 1;
		binary_transform <<<block, thread>>> (N, d_S, binary_threshold);		// binaryzation

		cudaMemcpy(h_S, d_S, N * sizeof(unsigned char), cudaMemcpyDeviceToHost);

		S.copy(h_S);
#endif
	}

	// get the vexel and image stack size
	dx = (float)args["stackres"].as_float(0);
	dy = (float)args["stackres"].as_float(1);
	dz = (float)args["stackres"].as_float(2);

	// get the save directory of image stack
	if (args["stackdir"].is_set())
		stackdir = args["stackdir"].as_string();

	// get the scale-down factor is provided
	if (args["scale"].is_set())
		scale = (float)args["scale"].as_float();

	// glut main loop
	bb = flow.boundingbox();
	glut_initialize();
	glutMainLoop();
}
