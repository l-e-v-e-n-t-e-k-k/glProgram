//=============================================================================================
// Zöld háromszög: A framework.h osztályait felhasználó megoldás
//=============================================================================================
#include "framework.h"
#include <iostream>
#include <cmath>
#include <memory>
#include <vector>
#include <string>

// pixel árnyaló
const char* fragSource = R"(
     #version 330 core
in vec3 wPos;
    in vec3 wNormal;
    in vec3 wView;
    in vec3 wLight;
    in vec2 TexCoord_fs;
    out vec4 fragmentColor;

    uniform vec3 ka;
    uniform vec3 kd;
    uniform vec3 ks;
    uniform float shine;
 uniform bool u_debugShadow;
 uniform vec4 wLiPos;

    uniform vec3 La;
    uniform vec3 Le;

    uniform bool u_isCheckerboard;
    uniform vec3 u_blueDiffuseColor;
    uniform vec3 u_whiteDiffuseColor;

uniform int triangleCount;
uniform vec3 triangles[768]; // max 256 haromszog

bool intersectTriangle(vec3 start, vec3 dir, vec3 r1, vec3 r2, vec3 r3) {
    vec3 n = cross(r2 - r1, r3 - r1); // normálvektor
    float denom = dot(dir, n);
    
    // Sugár párhuzamos a síkkal?
    if (abs(denom) < 1e-6) return false;

    float t = dot(r1 - start, n) / denom;

    // A metszéspont hátrafelé van?
    if (t < 0.0) return false;

    vec3 p = start + dir * t;

    // Háromszögön belül van-e a metszéspont?
    vec3 edge1 = r2 - r1;
    vec3 vp1 = p - r1;
    if (dot(cross(edge1, vp1), n) < 0.0) return false;

    vec3 edge2 = r3 - r2;
    vec3 vp2 = p - r2;
    if (dot(cross(edge2, vp2), n) < 0.0) return false;

    vec3 edge3 = r1 - r3;
    vec3 vp3 = p - r3;
    if (dot(cross(edge3, vp3), n) < 0.0) return false;

    return true;
}

bool isInShadow(vec3 point, vec3 L) {
    vec3 rayStart = point;

    for (int i = 0; i < triangleCount; i++) {
int idx = i*3;
        vec3 v0 = triangles[idx];
        vec3 v1 = triangles[idx+1];
        vec3 v2 = triangles[idx+2];
        
        if (intersectTriangle(rayStart, L, v0, v1, v2)) {
            return true;
        }
    }
    return false;
}

void main() {
    vec3 N = normalize(wNormal);
    vec3 V = normalize(wView);
    vec3 L = normalize(wLiPos.xyz);
    vec3 H = normalize(L + V);

 

   //  fragmentColor = vec4(normalize(wNormal) * 0.5 + 0.5, 1.0);
   // return;
    
    // Debug: feny irany
    // fragmentColor = vec4(normalize(wLight) * 0.5 + 0.5, 1.0);
   //  return;
    
    // Debug: haromszogek
   //  if (triangleCount > 0) {
   //      fragmentColor = vec4(1,0,0,1); // red if triangles exist
  //   } else {
   //      fragmentColor = vec4(0,1,0,1); // green if no triangles
  //   }
   //  return;
    float cosThetaLight = max(dot(N, L), 0.0);
    float cosThetaHalf = max(dot(N, H), 0.0);

 vec3 matKd;
    if (u_isCheckerboard) {
        float sum_coords_floor = floor(TexCoord_fs.x) + floor(TexCoord_fs.y);
        if (mod(sum_coords_floor, 2.0) == 0.0) {
            matKd = u_whiteDiffuseColor;
        } else {
            matKd = u_blueDiffuseColor;
        }
            if (isInShadow(wPos, L)) {
                matKd *= 0.5; // Árnyékos részek
            }
    } else {
        matKd = kd;
    }

if (u_debugShadow) {
        bool shadowed = isInShadow(wPos, L);
        
        if (shadowed) {
            // Árnyékban - piros 
            float ambient = 0.3;
            fragmentColor = vec4(0.0, 0.0, 0.0, 1.0);
        } else {
            // Megvilágított
            float diff = max(dot(normalize(wNormal), L), 0.0);
            fragmentColor = vec4(1.0, 1.0, 1.0, 1.0);
        }
        return;
    }

    vec3 Color;
    if (isInShadow(wPos, L)) {
        Color = ka * La; // csak ambient
    } else {
        vec3 ambient = ka * La;
        vec3 diffuse = matKd * cosThetaLight * Le;
        vec3 specular = vec3(0.0);
        if (cosThetaLight > 0.0) {
            specular = ks * pow(cosThetaHalf, shine) * Le;
        }
        Color = ambient + diffuse + specular;
    }

    Color = clamp(Color, 0.0f, 1.0f);

    fragmentColor = vec4(Color, 1.0);
}
)";

// csúcspont árnyaló
const char* vertSource = R"(
    #version 330 core
out vec3 wPos;
    uniform mat4 MVP, M, Minv; // MVP, Model, Model-inverse
    uniform vec4 wLiPos; // pos of light source
    uniform vec3 wEye; // pos of eye
    layout(location = 0) in vec3 vtxPos; // pos in model sp
    layout(location = 1) in vec3 vtxNorm;// normal in model sp
    out vec3 wNormal; // normal in world space
    out vec3 wView; // view in world space
    out vec3 wLight; // light dir in world space

        layout (location = 2) in vec2 vtxTexCoord;
        out vec2 TexCoord_fs;

    void main() {
     gl_Position = MVP * vec4(vtxPos, 1); // NDC
             vec4 wPos = M * vec4(vtxPos, 1);
             wLight = wLiPos.xyz;
             wView = wEye - wPos.xyz/wPos.w;
             wNormal = (vec4(vtxNorm, 0) * Minv).xyz;

             TexCoord_fs = vtxTexCoord;
    }

)";
inline vec3 myAbs(const vec3& v) {
    return vec3(std::fabs(v.x), std::fabs(v.y), std::fabs(v.z));
}

inline vec3 perpendicular(const vec3& v_in) {
    vec3 v = normalize(v_in);

    vec3 abs_v = myAbs(v);
    vec3 axis = (abs_v.x < abs_v.y) ?
        ((abs_v.x < abs_v.z) ? vec3(1, 0, 0) : vec3(0, 0, 1)) :
        ((abs_v.y < abs_v.z) ? vec3(0, 1, 0) : vec3(0, 0, 1));

    return normalize(cross(v, axis));
}

inline vec3 clamp(const vec3& v, float minVal, float maxVal) {
    return vec3(
        std::fmax(minVal, std::fmin(v.x, maxVal)),
        std::fmax(minVal, std::fmin(v.y, maxVal)),
        std::fmax(minVal, std::fmin(v.z, maxVal))
    );
}

inline vec3 pow(const vec3& base, const vec3& exp) {
    return vec3(
        std::pow(base.x, exp.x),
        std::pow(base.y, exp.y),
        std::pow(base.z, exp.z)
    );
}

inline float lengthSqr(const vec3& v) {
    return v.x * v.x + v.y * v.y + v.z * v.z;
}

class Material {
public:
    vec3 n;
    vec3 kappa;

    vec3 ka;
    vec3 kd;
    vec3 ks;
    float shine;

    bool isReflective;
    bool isRefractive;
    bool isRough;

    Material(float n_, float kappa_,
        const vec3& ka_, const vec3& kd_, const vec3& ks_,
        float shine_,
        bool reflective, bool refractive, bool rough)
        : n(vec3(n_)), kappa(vec3(kappa_)), ka(ka_), kd(kd_), ks(ks_), shine(shine_),
        isReflective(reflective), isRefractive(refractive), isRough(rough) {
    }

    Material(const vec3& n_, const vec3& kappa_,
        const vec3& ka_, const vec3& kd_, const vec3& ks_,
        float shine_,
        bool reflective, bool refractive, bool rough)
        : n(n_), kappa(kappa_), ka(ka_), kd(kd_), ks(ks_), shine(shine_),
        isReflective(reflective), isRefractive(refractive), isRough(rough) {
    }

    vec3 FresnelRGB(const vec3& incident, const vec3& normal) const {
        float cosTheta = std::max(0.0f, -dot(incident, normal));

        vec3 R0;

        if (lengthSqr(kappa) > 1e-8f) {
            vec3 n_plus_1 = n + vec3(1.0f);
            vec3 n_minus_1 = n - vec3(1.0f);
            vec3 k_sqr = kappa * kappa;

            vec3 numerator = (n_minus_1 * n_minus_1 + k_sqr);
            vec3 denominator = (n_plus_1 * n_plus_1 + k_sqr);

            R0 = vec3(numerator.x / denominator.x,
                numerator.y / denominator.y,
                numerator.z / denominator.z);
        }
        else {
            vec3 numerator = n - vec3(1.0f);
            vec3 denominator = n + vec3(1.0f);

            vec3 term = vec3(numerator.x / denominator.x,
                numerator.y / denominator.y,
                numerator.z / denominator.z);

            R0 = term * term;
        }

        return R0 + (vec3(1.0f) - R0) * std::pow(1.0f - cosTheta, 5.0f);
    }

    bool refract(const vec3& incident, const vec3& normal, float eta, vec3& refractedDir) const {
        float cos_i = -dot(incident, normal);

        float sin2_i = 1.0f - cos_i * cos_i;
        float sin2_t = eta * eta * sin2_i;

        if (sin2_t > 1.0f || sin2_t < -1e-6f) {
            return false;
        }

        float cos_t = sqrt(std::max(0.0f, 1.0f - sin2_t));
        refractedDir = normalize(eta * incident + (eta * cos_i - cos_t) * normal);
        return true;
    }

    vec3 reflect(const vec3& incident, const vec3& normal) const {
        return incident - 2.0f * dot(incident, normal) * normal;
    }

    vec3 shade(const vec3& lightDir, const vec3& viewDir, const vec3& normal, const vec3& lightColor) const {
        float diff = fmax(dot(lightDir, normal), 0.0f);
        vec3 diffuse = kd * diff * lightColor;

        vec3 lightReflectDir = reflect(-lightDir, normal);
        float spec = std::pow(fmax(dot(viewDir, lightReflectDir), 0.0f), shine);
        vec3 specular = ks * spec * lightColor;

        vec3 result = diffuse + specular;
        return result;
    }
};

class Camera {
private:
    vec3 eye;
    vec3 lookat;
    vec3 up_initial;
    vec3 up;
    vec3 right;
    float fov;
    float aspect;
    float fov_degrees = 45.0f;
    float near_plane = 0.1f; // Közelvágósík
    float far_plane = 100.0f; // Távolvágósík

    float orbitAngle;
    float orbitRadius;
    vec3 orbitCenter;

    void updateVectors() {
        vec3 forward = normalize(lookat - eye);
        right = normalize(cross(forward, up_initial));
        up = normalize(cross(right, forward));
    }

public:
    Camera(const vec3& eye_ = vec3(0, 1, 4), const vec3& lookat_ = vec3(0, 0, 0), const vec3& up_param = vec3(0, 1, 0), float fov_degrees = 45.0f, float aspect_ = 1.0f)
        : eye(eye_), lookat(lookat_), up_initial(normalize(up_param)), aspect(aspect_) {

        fov = fov_degrees * M_PI / 180.0f;

        orbitCenter = vec3(0, eye.y, 0);

        vec3 r_vec_xz = vec3(eye.x - orbitCenter.x, 0, eye.z - orbitCenter.z);

        orbitRadius = length(r_vec_xz);
        if (orbitRadius > 1e-5) {
            orbitAngle = atan2(eye.z - orbitCenter.z, eye.x - orbitCenter.x);
        }
        else {
            orbitAngle = 0;
        }

        updateVectors();
    }

    void moveOnOrbit() {
        orbitAngle += M_PI / 4.0f;

        eye.x = orbitCenter.x + orbitRadius * cos(orbitAngle);
        eye.z = orbitCenter.z + orbitRadius * sin(orbitAngle);
        eye.y = orbitCenter.y;

        updateVectors();

    }

    const vec3& getEye() const { return eye; }
    void setEye(const vec3& newEye) { eye = newEye; updateVectors(); }
    void setLookAt(const vec3& newLookAt) { lookat = newLookAt; updateVectors(); }
    void setAspectRatio(float asp) { aspect = asp; }
    void setFovDegrees(float degrees) { fov = degrees * M_PI / 180.0f; }
    void setNearFar(float n, float f) { near_plane = n; far_plane = f; }

    mat4 V() const { return lookAt(eye, lookat, up); } // Marked as const
    mat4 P() const { return perspective(fov, aspect, near_plane, far_plane); } // Marked as const

};

struct Light {
    vec3 direction;
    vec3 color;
    float intensity;

    Light(const vec3& dir, float intensity = 2.0f, const vec3& color = vec3(0.7f))
        : direction(normalize(dir)), color(color), intensity(intensity) {
    }

    vec3 getLightDir(const vec3& hitPos) const {
        return -direction;
    }

    vec3 getInRad() const {
        return color * intensity;
    }
    vec3 getAmbientRadianceContribution() const {
        return color * intensity * 0.2f;
    }
};



const int winWidth = 600, winHeight = 600;

struct VtxData {
    vec3 pos;
    vec3 normal;
    vec2 texcoord;
};


class Object3D : public Geometry<VtxData> {
    int nVtxInStrip, nStrips;
public:
    Object3D() {
        glEnableVertexAttribArray(0); // 0. regiszter = pozíció
        glEnableVertexAttribArray(1); // 1. regiszter = normál vektor
        glEnableVertexAttribArray(2); // 2. regiszter = textúra koordináta
        int nb = sizeof(VtxData);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, nb, (void*)offsetof(VtxData, pos));
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, nb, (void*)offsetof(VtxData, normal));
        glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, nb, (void*)offsetof(VtxData, texcoord));
    }

    virtual VtxData GenVtxData(float u, float v) = 0;


    void create(int M, int N) { // j == u, i+1 == v
        vtx.clear();
        nVtxInStrip = (M + 1) * 2;
        nStrips = N;
        for (int i = 0; i < N; i++) {
            for (int j = 0; j <= M; j++) {
                vtx.push_back(GenVtxData((float)j / M, (float)i / N));
                vtx.push_back(GenVtxData((float)j / M, (float)(i + 1) / N));
            }
        }
        updateGPU();
    }
    void Draw() {
        Bind();
        for (unsigned int i = 0; i < nStrips; i++)
            glDrawArrays(GL_TRIANGLE_STRIP, i * nVtxInStrip, nVtxInStrip);
    }
    virtual ~Object3D() {}

    const std::vector<VtxData>& getVertexData() const { return vtx; }
    int getNumStrips() const { return nStrips; }
    int getVerticesPerStrip() const { return nVtxInStrip; }

};

class Cylinder : public Object3D {
    vec3 base, axis;
    float radius, height;
    //Material material;

public:
    Cylinder(const vec3& centerOfBase, const vec3& directionOfAxis, float rad, float h)
        : base(centerOfBase), axis(normalize(directionOfAxis)), radius(rad), height(h) {
        this->create(6, 1);
        std::cout << "Cylinder created with " << vtx.size() << " vertices." << std::endl;
    }

    VtxData GenVtxData(float u, float v) override { // v magasság u szög
        VtxData vtx;
        float theta = u * 2 * M_PI;

        // Alap és tetejének pozíciója
        vec3 bottom = base;
        vec3 top = base + axis * height;

        vec3 radialDir = perpendicular(axis);
        vec3 tangentDir = cross(axis, radialDir);
        vec3 pos = bottom +
            (cosf(theta) * radialDir + sinf(theta) * cross(axis, radialDir)) * radius +
            axis * v * height;

        vtx.pos = pos;
        vtx.normal = normalize(pos - (bottom + axis * v * height));
        vtx.texcoord = vec2(u, v);
        return vtx;
    }
};

class Cone : public Object3D {
    vec3 p, d; // p = csúcs, d = irány (normalizált)
    float alpha, h; // aplha = A kúp félnyílásszöge (alpha radiánban!) h = Magasság (h)

public:
    Cone(const vec3& apex, const vec3& axisDirection, float halfAngleRad, float h)
        : p(apex), d(normalize(axisDirection)), alpha(halfAngleRad), h(h) {
        this->create(6, 1);
        std::cout << "Cone created with " << vtx.size() << " vertices." << std::endl;
    }

    VtxData GenVtxData(float u, float v) override {
        v *= 0.99f;
        VtxData vtx;
        float theta = u * 2 * M_PI;
        float currentHeight = v * h;
        float currentRadius = tanf(alpha) * v * h;

        vec3 radialDir = perpendicular(d);
        vec3 tangentDir = cross(d, radialDir);


        vec3 circOffset = (radialDir * cosf(theta) + tangentDir * sinf(theta)) * currentRadius;
        vtx.pos = p + d * currentHeight + circOffset;
        if (currentRadius < 0.0001f) {
            vtx.normal = d;
        }
        else {
            vec3 radialNormalComp = normalize(circOffset);
            vtx.normal = normalize(radialNormalComp * cosf(alpha) - d * sinf(alpha));
        }

        vtx.texcoord = vec2(u, v);
        return vtx;
    }
};

class Plane : public Object3D {
    float y, size, tileSize;

public:
    Plane(float yPosition, float size, float tileSize)
        : y(yPosition), size(size), tileSize(tileSize) {
        this->create(1, 1);
        std::cout << "Plane created with " << vtx.size() << " vertices." << std::endl;
    }

    VtxData GenVtxData(float u, float v) override {
        VtxData vtx;
        float x = (u - 0.5f) * size;
        float z = (v - 0.5f) * size;

        vtx.pos = vec3(x, y, z);
        vtx.normal = vec3(0, 1, 0);
        vtx.texcoord = vec2(x / tileSize, z / tileSize);

        return vtx;
    }
};


struct Renderable {
public:
    Object3D* geometry = nullptr;
    mat4 modelMatrix;
    Material material;

    Renderable(Object3D* geom, const mat4& modelMat, const Material& mat)
        : geometry(geom), modelMatrix(modelMat), material(mat) {
    }

};

class Scene {
private:
    std::vector<Renderable> renderables;
    Light light;
    vec3 ambientLight = vec3(0.4f);

public:
    Camera camera;


    Scene() : light(vec3(-1, -1, -1), 2.0f) {
    }

    ~Scene() {
        for (Renderable& r : renderables) {
            if (r.geometry) {
                delete r.geometry;
                r.geometry = nullptr;
            }
        }
        renderables.clear();
        std::cout << "Scene cleanup finished.\n";
    }

    std::vector<glm::vec3> extractWorldTriangles(const std::vector<Renderable>& renderables) const {
        std::vector<glm::vec3> shadowTriangles;

        for (const Renderable& r : renderables) {

            const auto& vertices = r.geometry->getVertexData();
            int stripCount = r.geometry->getNumStrips();
            int vtxPerStrip = r.geometry->getVerticesPerStrip();

            for (int s = 0; s < stripCount; s++) {
                int offset = s * vtxPerStrip;
                for (int i = 0; i < vtxPerStrip - 2; i++) {
                    // 1. poziciobol vec4
                    glm::vec4 pos0(vertices[offset + i].pos.x,
                        vertices[offset + i].pos.y,
                        vertices[offset + i].pos.z,
                        1.0f);

                    // 2. transzformaljuk a poziciot
                    glm::vec4 temp0 = r.modelMatrix * pos0;
                    // vec3ma
                    glm::vec3 v0(temp0.x, temp0.y, temp0.z);

                    glm::vec4 pos1(vertices[offset + i + 1].pos.x,
                        vertices[offset + i + 1].pos.y,
                        vertices[offset + i + 1].pos.z,
                        1.0f);
                    glm::vec4 temp1 = r.modelMatrix * pos1;
                    glm::vec3 v1(temp1.x, temp1.y, temp1.z);

                    glm::vec4 pos2(vertices[offset + i + 2].pos.x,
                        vertices[offset + i + 2].pos.y,
                        vertices[offset + i + 2].pos.z,
                        1.0f);
                    glm::vec4 temp2 = r.modelMatrix * pos2;
                    glm::vec3 v2(temp2.x, temp2.y, temp2.z);

                    shadowTriangles.push_back(v0);
                    shadowTriangles.push_back(v1);
                    shadowTriangles.push_back(v2);
                }
            }
        }
        return shadowTriangles;
    }

    void build() {

        std::cout << "Scene::build() called - creating only the specified gold cylinder." << std::endl;
        renderables.clear();
        std::cout << "Renderables cleared. Count: " << renderables.size() << std::endl;

        std::cout << " - Attempting to create gold cylinder (specified parameters)..." << std::endl;

        vec3 gold_n_original = vec3(0.17f, 0.35f, 1.5f);
        vec3 gold_kappa_original = vec3(3.1f, 2.7f, 1.9f);

        vec3 gold_kd_rough = gold_n_original;
        vec3 gold_ks_rough = gold_kappa_original;

        vec3 gold_ka_rough = gold_n_original * 3.0f;

        float gold_shine_rough = 200.0f;

        Material rough_gold_material(
            vec3(0.0f),
            vec3(0.0f),
            gold_ka_rough,
            gold_kd_rough,
            gold_ks_rough,
            gold_shine_rough,
            false,
            false,
            true
        );
        std::cout << "   Gold material (Phong compatible) created." << std::endl;

        Cylinder* goldCylinderGeom = nullptr;

        try {
            std::cout << "   Instantiating Cylinder object with specified parameters..." << std::endl;
            vec3 cylinderBase = vec3(1.0f, -1.0f, 0.0f);
            vec3 cylinderAxis = vec3(0.1f, 1.0f, 0.0f);
            float cylinderRadius = 0.3f;
            float cylinderHeight = 2.0f;

            goldCylinderGeom = new Cylinder(cylinderBase, cylinderAxis, cylinderRadius, cylinderHeight);

            if (goldCylinderGeom) {
                std::cout << "   Cylinder object instantiated successfully." << std::endl;
            }
            else {
                std::cerr << "   ERROR: new Cylinder returned nullptr!" << std::endl;
            }

            mat4 goldCylinderModelMatrix;
            for (int i = 0; i < 4; ++i) { for (int j = 0; j < 4; ++j) { goldCylinderModelMatrix[i][j] = 0.0f; } }
            goldCylinderModelMatrix[0][0] = 1.0f;
            goldCylinderModelMatrix[1][1] = 1.0f;
            goldCylinderModelMatrix[2][2] = 1.0f;
            goldCylinderModelMatrix[3][3] = 1.0f;

            std::cout << "   Model matrix (identity) created." << std::endl;

            if (goldCylinderGeom) {
                std::cout << "   Adding Renderable to vector..." << std::endl;
                renderables.emplace_back(goldCylinderGeom, goldCylinderModelMatrix, rough_gold_material);
                std::cout << "   Renderable added. Renderables count: " << renderables.size() << std::endl;
            }
            else {
                std::cerr << "   Skipping adding Renderable because Cylinder geometry is null." << std::endl;
            }

        }
        catch (const std::exception& e) {
            std::cerr << "   EXCEPTION during Cylinder creation or adding: " << e.what() << std::endl;
            delete goldCylinderGeom;
            goldCylinderGeom = nullptr;
        }

        std::cout << " - Attempting to create yellow plastic cylinder..." << std::endl;

        vec3 yellow_kd = vec3(0.3f, 0.2f, 0.1f);
        vec3 yellow_ks = vec3(2.0f, 2.0f, 2.0f);
        float yellow_shininess = 50.0f;
        vec3 yellow_ka = yellow_kd * 3.0f;


        Material yellow_plastic_material(
            vec3(0.0f),
            vec3(0.0f),
            yellow_ka,
            yellow_kd,
            yellow_ks,
            yellow_shininess,
            false,
            false,
            true
        );
        std::cout << "   Yellow plastic material created." << std::endl;


        Cylinder* plasticCylinderGeom = nullptr;

        try {
            std::cout << "   Instantiating Cylinder object with specified parameters..." << std::endl;
            vec3 cylinderBase = vec3(-1.0f, -1.0f, 0.0f);
            vec3 cylinderAxis = vec3(0.0f, 1.0f, 0.1f);
            float cylinderRadius = 0.3f;
            float cylinderHeight = 2.0f;


            plasticCylinderGeom = new Cylinder(cylinderBase, cylinderAxis, cylinderRadius, cylinderHeight);

            if (plasticCylinderGeom) {
                std::cout << "   Cylinder object instantiated successfully." << std::endl;
            }
            else {
                std::cerr << "   ERROR: new Cylinder returned nullptr!" << std::endl;
            }

            mat4 plasticCylinderModelMatrix;
            for (int i = 0; i < 4; ++i) { for (int j = 0; j < 4; ++j) { plasticCylinderModelMatrix[i][j] = 0.0f; } }
            plasticCylinderModelMatrix[0][0] = 1.0f;
            plasticCylinderModelMatrix[1][1] = 1.0f;
            plasticCylinderModelMatrix[2][2] = 1.0f;
            plasticCylinderModelMatrix[3][3] = 1.0f;
            std::cout << "   Model matrix (identity) created." << std::endl;


            if (plasticCylinderGeom) {
                std::cout << "   Adding Renderable to vector..." << std::endl;
                renderables.emplace_back(plasticCylinderGeom, plasticCylinderModelMatrix, yellow_plastic_material);
                std::cout << "   Renderable added. Renderables count: " << renderables.size() << std::endl;
            }
            else {
                std::cerr << "   Skipping adding Renderable because Cylinder geometry is null." << std::endl;
            }

        }
        catch (const std::exception& e) {
            std::cerr << "   EXCEPTION during Cylinder creation or adding: " << e.what() << std::endl;
            delete plasticCylinderGeom;
            plasticCylinderGeom = nullptr;
        }

        std::cout << " - Creating checkerboard plane as specified..." << std::endl;

        Material checkerboardPlaneMaterial(
            vec3(0.0f),
            vec3(0.0f),
            vec3(0.05f, 0.05f, 0.05f),
            vec3(0.3f, 0.3f, 0.3f),
            vec3(0.0f, 0.0f, 0.0f),
            1.0f,
            false, false, true
        );
        std::cout << "   Checkerboard plane material template created." << std::endl;

        float planeYPosition = -1.0f;
        float planeSquareSize = 20.0f;
        float csempeMeret = 1.0f;

        Plane* planeGeometry = nullptr;
        try {
            planeGeometry = new Plane(planeYPosition, planeSquareSize, csempeMeret);
            if (planeGeometry) {
                std::cout << "   Plane geometry instantiated (y=" << planeYPosition << ", size=" << planeSquareSize << ", tileSize=" << csempeMeret << ")." << std::endl;
            }
            else {
                std::cerr << "   ERROR: new Plane returned nullptr!" << std::endl;
            }

            mat4 planeModelMatrix;
            for (int r = 0; r < 4; ++r) for (int c = 0; c < 4; ++c) planeModelMatrix[c][r] = 0.0f;
            planeModelMatrix[0][0] = 1.0f; planeModelMatrix[1][1] = 1.0f;
            planeModelMatrix[2][2] = 1.0f; planeModelMatrix[3][3] = 1.0f;
            std::cout << "   Plane model matrix (identity) created." << std::endl;

            if (planeGeometry) {
                renderables.emplace_back(planeGeometry, planeModelMatrix, checkerboardPlaneMaterial);
                std::cout << "   Checkerboard Plane Renderable added. Renderables count: " << renderables.size() << std::endl;
            }

        }
        catch (const std::exception& e) {
            std::cerr << "   EXCEPTION during Plane creation or adding: " << e.what() << std::endl;
            delete planeGeometry;
            planeGeometry = nullptr;
        }

        std::cout << " - Attempting to create ROUGH water cylinder..." << std::endl;

        vec3 water_n_original = vec3(1.3f);
        vec3 water_kappa_original = vec3(0.0f);

        vec3 water_kd_rough = water_n_original;
        vec3 water_ka_rough = water_kd_rough * 3.0f;
        vec3 water_ks_rough = water_kappa_original;
        float water_shine_rough = 5.0f;

        Material rough_water_material(
            vec3(0.0f),
            vec3(0.0f),
            water_ka_rough,
            water_kd_rough,
            water_ks_rough,
            water_shine_rough,
            false,
            false,
            true
        );


        std::cout << "   Rough water material created." << std::endl;

        Cylinder* waterCylinderGeom = nullptr;

        try {
            std::cout << "   Instantiating Cylinder object (water)..." << std::endl;

            vec3 cylinderBase = vec3(0.0f, -1.0f, -0.8f);
            vec3 cylinderAxis = vec3(-0.2f, 1.0f, -0.1f);
            float cylinderRadius = 0.3f;
            float cylinderHeight = 2.0f;

            waterCylinderGeom = new Cylinder(cylinderBase, cylinderAxis, cylinderRadius, cylinderHeight);

            if (waterCylinderGeom) {
                std::cout << "   Cylinder object (water) instantiated successfully." << std::endl;
            }
            else {
                std::cerr << "   ERROR: new Cylinder (water) returned nullptr!" << std::endl;
            }

            mat4 waterCylinderModelMatrix;
            for (int r = 0; r < 4; ++r) for (int c = 0; c < 4; ++c) waterCylinderModelMatrix[c][r] = 0.0f;
            waterCylinderModelMatrix[0][0] = 1.0f; waterCylinderModelMatrix[1][1] = 1.0f;
            waterCylinderModelMatrix[2][2] = 1.0f; waterCylinderModelMatrix[3][3] = 1.0f;

            std::cout << "   Model matrix (identity) for water cylinder created." << std::endl;


            if (waterCylinderGeom) {
                std::cout << "   Adding Renderable (water) to vector..." << std::endl;
                renderables.emplace_back(waterCylinderGeom, waterCylinderModelMatrix, rough_water_material);
                std::cout << "   Renderable (water) added. Renderables count: " << renderables.size() << std::endl;
            }
            else {
                std::cerr << "   Skipping adding Renderable (water) because Cylinder geometry is null." << std::endl;
            }

        }
        catch (const std::exception& e) {
            std::cerr << "   EXCEPTION during Cylinder (water) creation or adding: " << e.what() << std::endl;
            delete waterCylinderGeom;
            waterCylinderGeom = nullptr;
        }

        std::cout << "Scene::build() called - creating cyan and magenta cones." << std::endl;


        // CIÁN KÚP 
        std::cout << " - Attempting to create cyan cone..." << std::endl;

        vec3 cyan_kd = vec3(0.1f, 0.2f, 0.3f);
        vec3 cyan_ks = vec3(2.0f, 2.0f, 2.0f);
        float cyan_shininess = 100.0f;
        vec3 cyan_ka = cyan_kd * 3.0f;

        Material cyan_cone_material(
            vec3(1.5f),
            vec3(0.0f),
            cyan_ka,
            cyan_kd,
            cyan_ks,
            cyan_shininess,
            false, false, true
        );
        std::cout << "   Cyan cone material created." << std::endl;

        Cone* cyanConeGeom = nullptr;
        try {
            std::cout << "   Instantiating Cone object (cyan)..." << std::endl;
            vec3 cyan_apex = vec3(0.0f, 1.0f, 0.0f);
            vec3 cyan_axis = vec3(-0.1f, -1.0f, -0.05f);
            float cone_half_angle = 0.2f;
            float cone_height = 2.0f;


            cyanConeGeom = new Cone(cyan_apex, cyan_axis, cone_half_angle, cone_height);

            if (cyanConeGeom) {
                std::cout << "   Cone object (cyan) instantiated successfully." << std::endl;
            }
            else {
                std::cerr << "   ERROR: new Cone (cyan) returned nullptr!" << std::endl;
            }


            mat4 cyanConeModelMatrix;
            for (int r_idx = 0; r_idx < 4; ++r_idx) for (int c_idx = 0; c_idx < 4; ++c_idx) cyanConeModelMatrix[c_idx][r_idx] = 0.0f;
            cyanConeModelMatrix[0][0] = 1.0f; cyanConeModelMatrix[1][1] = 1.0f;
            cyanConeModelMatrix[2][2] = 1.0f; cyanConeModelMatrix[3][3] = 1.0f;
            std::cout << "   Model matrix (identity) for cyan cone created." << std::endl;

            if (cyanConeGeom) {
                renderables.emplace_back(cyanConeGeom, cyanConeModelMatrix, cyan_cone_material);
                std::cout << "   Renderable (cyan cone) added. Renderables count: " << renderables.size() << std::endl;
            }

        }
        catch (const std::exception& e) {
            std::cerr << "   EXCEPTION during Cone (cyan) creation or adding: " << e.what() << std::endl;
            delete cyanConeGeom;
            cyanConeGeom = nullptr;
        }


        //  MAGENTA KÚP
        std::cout << " - Attempting to create magenta cone..." << std::endl;

        vec3 magenta_kd = vec3(0.3f, 0.0f, 0.2f);
        vec3 magenta_ks = vec3(2.0f, 2.0f, 2.0f);
        float magenta_shininess = 20.0f;
        vec3 magenta_ka = magenta_kd * 3.0f;

        Material magenta_cone_material(
            vec3(1.5f), vec3(0.0f),
            magenta_ka,
            magenta_kd,
            magenta_ks,
            magenta_shininess,
            false, false, true
        );
        std::cout << "   Magenta cone material created." << std::endl;

        Cone* magentaConeGeom = nullptr;
        try {
            std::cout << "   Instantiating Cone object (magenta)..." << std::endl;
            vec3 magenta_apex = vec3(0.0f, 1.0f, 0.8f);
            vec3 magenta_axis = vec3(0.2f, -1.0f, 0.0f);

            float cone_half_angle = 0.2f;
            float cone_height = 2.0f;

            magentaConeGeom = new Cone(magenta_apex, magenta_axis, cone_half_angle, cone_height);

            if (magentaConeGeom) {
                std::cout << "   Cone object (magenta) instantiated successfully." << std::endl;
            }
            else {
                std::cerr << "   ERROR: new Cone (magenta) returned nullptr!" << std::endl;
            }

            mat4 magentaConeModelMatrix;
            for (int r_idx = 0; r_idx < 4; ++r_idx) for (int c_idx = 0; c_idx < 4; ++c_idx) magentaConeModelMatrix[c_idx][r_idx] = 0.0f;
            magentaConeModelMatrix[0][0] = 1.0f; magentaConeModelMatrix[1][1] = 1.0f;
            magentaConeModelMatrix[2][2] = 1.0f; magentaConeModelMatrix[3][3] = 1.0f;
            std::cout << "   Model matrix (identity) for magenta cone created." << std::endl;

            if (magentaConeGeom) {
                renderables.emplace_back(magentaConeGeom, magentaConeModelMatrix, magenta_cone_material);
                std::cout << "   Renderable (magenta cone) added. Renderables count: " << renderables.size() << std::endl;
            }

        }
        catch (const std::exception& e) {
            std::cerr << "   EXCEPTION during Cone (magenta) creation or adding: " << e.what() << std::endl;
            delete magentaConeGeom;
            magentaConeGeom = nullptr;
        }


        std::cout << "Scene::build() finished. Total renderables: " << renderables.size() << std::endl;


    }

    mat4 manualInverse(const mat4& m) const {
        mat4 inv;
        // Forgatási rész transzponálása
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                inv[i][j] = m[j][i];
            }
        }
        // Eltolási rész
        vec3 t(m[3][0], m[3][1], m[3][2]); // Eltolás kinyerése
        inv[3][0] = -(inv[0][0] * t.x + inv[1][0] * t.y + inv[2][0] * t.z);
        inv[3][1] = -(inv[0][1] * t.x + inv[1][1] * t.y + inv[2][1] * t.z);
        inv[3][2] = -(inv[0][2] * t.x + inv[1][2] * t.y + inv[2][2] * t.z);

        // Utolsó sor és oszlop
        inv[0][3] = inv[1][3] = inv[2][3] = 0.0f;
        inv[3][3] = 1.0f;
        return inv;
    }



    void render(GPUProgram& shaderProgram) const {
        std::vector<vec3> shadowTriangles = extractWorldTriangles(renderables);
        std::cout << "Total triangles for shadow: " << shadowTriangles.size() / 3 << std::endl;
        shaderProgram.Use(); // Shader program
        shaderProgram.setUniform(false, "u_debugShadow");
        int maxTriangles = 256;
        int triangleCount = fmin<int>(shadowTriangles.size() / 3, maxTriangles);

        shaderProgram.setUniform(triangleCount, "triangleCount");

        for (int i = 0; i < triangleCount * 3; ++i) {
            std::string name = "triangles[" + std::to_string(i) + "]";
            shaderProgram.setUniform(shadowTriangles[i], name);
        }

        // Kamera pozíciója
        shaderProgram.setUniform(camera.getEye(), "wEye");


        // Fény tulajdonságainak beállítása
        vec3 lightDirForUniform = -light.direction; // A fényforrás felé mutató irány
        vec4 lightPosOrDirUniform = vec4(lightDirForUniform.x, lightDirForUniform.y, lightDirForUniform.z, 0.0f);
        shaderProgram.setUniform(lightPosOrDirUniform, "wLiPos");

        // La: Ambient fény teljes sugárzása
        // Le: A fõ fényforrás diffúz és spekuláris komponensének sugárzása
        shaderProgram.setUniform(ambientLight, "La");
        shaderProgram.setUniform(light.getInRad(), "Le");

        vec3 specifiedBlueColor = vec3(0.0f, 0.1f, 0.3f);
        vec3 specifiedWhiteColor = vec3(0.75f, 0.75f, 0.75f);

        // Objektumok kirajzolása egyenként
        for (const Renderable& r : renderables) {
            if (!r.geometry) {
                continue;
            }

            // Modellspecifikus uniformok beállítása
            shaderProgram.setUniform(r.modelMatrix, "M");

            // MVP mátrix kiszámítása és beállítása
            mat4 mvpMatrix = camera.P() * camera.V() * r.modelMatrix;
            shaderProgram.setUniform(mvpMatrix, "MVP");

            // Minv mátrix a normálvektorok transzformálásához
            mat4 modelInverseMatrix = manualInverse(r.modelMatrix);
            shaderProgram.setUniform(modelInverseMatrix, "Minv");

            // Anyagspecifikus uniformok beállítása
            shaderProgram.setUniform(r.material.ka, "ka");
            shaderProgram.setUniform(r.material.kd, "kd");
            shaderProgram.setUniform(r.material.ks, "ks");
            shaderProgram.setUniform(r.material.shine, "shine");

            Plane* planeGeom = dynamic_cast<Plane*>(r.geometry);
            if (planeGeom) {
                std::cout << "Rendering Plane, setting u_isCheckerboard to true" << std::endl;
                shaderProgram.setUniform(true, "u_isCheckerboard");
                shaderProgram.setUniform(specifiedBlueColor, "u_blueDiffuseColor");
                shaderProgram.setUniform(specifiedWhiteColor, "u_whiteDiffuseColor");
            }
            else {
                shaderProgram.setUniform(false, "u_isCheckerboard");

            }

            r.geometry->Draw();
        }
    }
};

class Inkrementalis : public glApp {
    GPUProgram gpuProgram;
    Scene scene;
public:
    Inkrementalis() : glApp("Hazi5") {

    }

    void onInitialization() {

        glViewport(0, 0, winWidth, winHeight);
        glEnable(GL_DEPTH_TEST);
        glClearColor(0.3f, 0.3f, 0.3f, 1.0f);

        gpuProgram.create(vertSource, fragSource);
        if (!gpuProgram.link()) {
            std::cerr << "Shader program linking failed!" << std::endl;
            exit(-1);
        }

        scene.build();

        std::cout << "Initialization complete\n";
    }

    void onDisplay() override {

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        scene.render(gpuProgram);

    }

    void onKeyboard(int key) override {
        if (key == 'a' || key == 'A') {
            std::cout << "\n'a' key pressed - moving camera.\n";
            scene.camera.moveOnOrbit();

            refreshScreen();
        }
    }
};

Inkrementalis app;