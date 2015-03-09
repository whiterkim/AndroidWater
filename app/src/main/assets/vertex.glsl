#version 300 es

uniform mat4 MVP;
uniform vec2 size;

in vec3 vertex_position;
in vec3 vertex_normal;

out vec3 normal;
out vec2 uv;
out vec3 vertex_position_w;

void main(){
  vertex_position_w = vertex_position;
  gl_Position =  MVP * vec4(vertex_position, 1.0);
  normal = vertex_normal;
  uv = vertex_position.xy / (size - vec2(1.0));
}