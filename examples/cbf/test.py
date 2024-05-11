import pathlib as p

import numpy as np
import pinocchio as pin
from robot_descriptions.panda_description import URDF_PATH

urdf_path = p.Path(URDF_PATH)
filename = str(urdf_path)
print(filename)

# Load model
model = pin.buildModelFromUrdf(filename)

# Load collision geometries
geom_model = pin.buildGeomFromUrdf(
    model,
    filename,
    str(urdf_path.parents[4]),
    pin.GeometryType.COLLISION,
)

# Add collisition pair
geom_model.addAllCollisionPairs()
for i in range(model.nq):
    geom_model.removeCollisionPair(pin.CollisionPair(i, i + 1))
print("num collision pairs - initial:", len(geom_model.collisionPairs))

q = np.random.rand(model.nq)

# Create data structures
data = model.createData()
geom_data = pin.GeometryData(geom_model)

# Compute all the collisions
pin.computeCollisions(model, data, geom_model, geom_data, q, False)
pin.computeDistances(model, data, geom_model, geom_data, q)
N = len(geom_model.collisionPairs)
# Print the status of collision for all collision pairs
for k in range(len(geom_model.collisionPairs)):
    cr = geom_data.collisionResults[k]
    cp = geom_model.collisionPairs[k]
    dr = geom_data.distanceResults[k]

    go_1 = geom_model.geometryObjects[cp.first].name
    go_2 = geom_model.geometryObjects[cp.second].name
    if cr.isCollision():
        print(
            f"collision pair #{k}:",
            go_1,
            ",",
            go_2,
            "- collision:",
            "Yes" if cr.isCollision() else "No",
        )
        print("distance:", dr.min_distance)

# Compute for a single pair of collision

# Computing jacobian
pin.computeJointJacobians(model, data, q)

J = np.zeros((N, model.nq))
Jrow_q = np.zeros(model.nq)
Jrow_v = np.zeros(model.nv)

dr = geom_data.distanceResults[0]

print()
print("Computing jacobian")
for k in range(len(geom_model.collisionPairs)):
    cr = geom_data.collisionResults[k]
    cp = geom_model.collisionPairs[k]
    dr = geom_data.distanceResults[k]

    go_1 = geom_model.geometryObjects[cp.first]
    go_2 = geom_model.geometryObjects[cp.second]

    j1_id = go_1.parentJoint
    j2_id = go_2.parentJoint
    if j1_id - j2_id == 1:
        continue

    w1 = dr.getNearestPoint1()
    r1 = w1 - data.oMi[j1_id].translation
    ddr_dw1 = -dr.normal.reshape(1, -1)
    print(np.linalg.norm(dr.normal))
    dw1_dj1 = np.block([np.eye(3), -pin.skew(r1)])
    dj1_dq = pin.getJointJacobian(
        model, data, j1_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
    )
    Jrow_v = ddr_dw1 @ dw1_dj1 @ dj1_dq

    w2 = dr.getNearestPoint2()
    r2 = w2 - data.oMi[j2_id].translation
    ddr_dw2 = dr.normal.reshape(1, -1)
    dw2_dj2 = np.block([np.eye(3), -pin.skew(r2)])
    dj2_dq = pin.getJointJacobian(
        model, data, j2_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
    )
    Jrow_v += ddr_dw2 @ dw2_dj2 @ dj2_dq

    if cr.isCollision():
        print(f"Collision pair #{k}:", go_1.name, ",", go_2.name)

    J[k] = Jrow_v

# print("bad values in distance jacobian:")
# print("q =", q)
# for k in range(len(geom_model.collisionPairs)):
#     cr = geom_data.collisionResults[k]
#     cp = geom_model.collisionPairs[k]
#     dr = geom_data.distanceResults[k]
#     print("at row", k, ":", J[k])
#     print("distance:", dr.min_distance)
#     print(f"collision pair #{k}:", cp.first, ",", cp.second, "- collision:", "Yes" if cr.isCollision() else "No")
