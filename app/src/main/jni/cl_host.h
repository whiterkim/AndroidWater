//
//  cl_host.h
//  Rain
//
//  Created by hyspace on 3/4/15.
//  Copyright (c) 2015 hyspace. All rights reserved.
//

#ifndef __Rain__cl_host__
#define __Rain__cl_host__

#define CL_USE_DEPRECATED_OPENCL_1_1_APIS
//#define __CL_ENABLE_EXCEPTIONS
#include <stdlib.h>
#include <CL/cl.hpp>
#include <CL/opencl.h>
#include <GLES3/gl3.h>
#include <EGL/egl.h>
#include "cl_host.h"
#include "log.h"

int initGLObjects(GLuint, GLuint);

int initCL(std::string&, EGLContext&, EGLDisplay&);

int freeCL(void);

int recompute(GLuint, GLuint, GLfloat, GLfloat, GLfloat);


#endif /* defined(__Rain__cl_host__) */
