#version 300 es

uniform struct Light {
  vec3 direction;
  vec3 intensities;
} light;

uniform vec3 camera_position;
uniform vec2 size;

uniform sampler2D sc_texture;
uniform sampler2D caustic_texture;

in vec3 normal;
in vec2 uv;
in vec3 vertex_position_w;

out vec3 color;

precision highp float;

void main()
{
  const float ita = 1.33;
  // calculate offset of refraction
  vec3 n = normalize(normal);
  vec3 l1 = camera_position - vertex_position_w;
  vec3 e = normalize(l1);
  vec3 f = refract(e, n, ita);
  vec2 offset = f.xy * (vertex_position_w.z / f.z);
  offset = offset / (size - vec2(1.0));
  vec2 final_uv = uv + offset;

  // calculate specular
  vec3 r = reflect(light.direction, n);
  float c = max(dot(r, -e), 0.0);
  vec3 specular = light.intensities * pow(c, 200.0) * 0.3;

  // add caustic texture with screencapture texture
  if(final_uv.x < 0.0 || final_uv. y < 0.0 || final_uv.x > 1.0 || final_uv.y > 1.0){
    color = vec3(0.0, 0.0, 0.0);
  }else{
    vec2 final_uv_i = vec2(final_uv.x, 1.0 - final_uv.y);
    color = texture(sc_texture, final_uv_i ).xyz +
            vec3(texture(caustic_texture, final_uv).r - 0.6) +
            specular;
    //color = specular;
    //color = vec3((texture( caustic_texture, final_uv ).x),(texture( caustic_texture, final_uv ).x),(texture( caustic_texture, final_uv ).x));
  }
}
