import math, numpy as np, plotly.graph_objects as go
SQRT3 = 3**0.5

# ────────────────────────────────
# utilitaires communs
# ────────────────────────────────
def _oddr_to_xy(r, q, R):
    return R * SQRT3 * (q + 0.5 * (r & 1)), -R * 1.5 * r

def _tri_fan(center_idx, ring_inds):
    return [(center_idx, ring_inds[k], ring_inds[(k+1)%len(ring_inds)])
            for k in range(len(ring_inds))]

def _mesh(vertices, tris, color, flat=True):
    x, y, z = vertices.T
    i, j, k = zip(*tris)
    return go.Mesh3d(
        x=x, y=y, z=z, i=i, j=j, k=k,
        color=color, flatshading=flat,
        lighting=dict(ambient=.9, diffuse=.1, specular=.05, roughness=.9),
        showscale=False
    )

def _border_trace(top_vertices, color="#000000", width=1):
    """
    Retourne une trace Scatter3d qui relie les sommets passés en argument
    (un array shape (6, 3)) pour dessiner un contour fin.
    """
    # fermer le polygone → on répète le premier point à la fin
    ring = np.vstack([top_vertices, top_vertices[0]])
    return go.Scatter3d(
        x=ring[:,0], y=ring[:,1], z=ring[:,2],
        mode="lines",
        line=dict(color=color, width=width),
        hoverinfo="skip",
        showlegend=False
    )

# ────────────────────────────────
# fonction principale
# ────────────────────────────────
def hex_prism_traces(r, q, *,
                     R=20, H=3,
                     fill_color='black',
                     right_face_color='#BDBDBD',
                     bottom_face_color='#707070',
                     other_side_color='#909090',
                     n_stars=0):
    """
    Retourne une liste de traces Plotly (Mesh3d) :
    - hexagone extrudé base z=0 → top z=H
    - n_stars étoiles prismes blanches alignées (si n_stars > 0)

    Seuls les arguments ci-dessus sont exposés ; outer_ratio / thickness
    des étoiles sont volontairement *hardcodés* dans la fonction.
    """
    traces = []

    # ---- hexagone ----
    cx, cy = _oddr_to_xy(r, q, R)
    ring2d = np.array([[math.cos(math.radians(60*i+30)),
                        math.sin(math.radians(60*i+30))]
                       for i in range(6)])
    base = np.column_stack((cx + R*ring2d[:,0],
                            cy + R*ring2d[:,1],
                            np.zeros(6)))
    top  = base.copy(); top[:,2] = H

    # faces sup/inf
    verts_top = np.vstack([[cx, cy, H], top])
    traces.append(_mesh(verts_top, _tri_fan(0, range(1,7)), fill_color))
    for interm_h in range(0, H+1, 7):
        interm_top = top.copy() ; interm_top[:,2] = interm_h
        traces.append(_border_trace(interm_top, color="#000000", width=1))
    verts_bot = np.vstack([[cx, cy, 0], base])
    traces.append(_mesh(verts_bot, _tri_fan(0, range(6,0,-1)), bottom_face_color))

    # faces latérales
    def _side_col(vec):
        return right_face_color if vec[0] > .5 else \
               bottom_face_color if vec[1] < -.5 else other_side_color
    for i in range(6):
        j = (i+1)%6
        quad = np.array([base[i], base[j], top[j], top[i]])
        col  = _side_col(base[j,:2] - base[i,:2])
        traces.append(_mesh(quad, [(0,1,2),(0,2,3)], col))

    # ---- étoiles (optionnel) ----
    if n_stars:
        outer_ratio, thickness, star_color = 0.30, 1, 'white'
        outer_r = R * outer_ratio
        inner_r = outer_r * 0.5
        spread  = R * 0.5
        xs = [0.] if n_stars == 1 else np.linspace(-spread, spread, n_stars)

        for dx in xs:
            cx_s = cx + dx
            total, rot = 10, math.radians(90)
            top_z = H + 1e-3

            # anneau points étoile
            ring_top = np.array([[cx_s + (outer_r if k%2==0 else inner_r) * math.cos(2*math.pi*k/total + rot),
                                  cy + (outer_r if k%2==0 else inner_r) * math.sin(2*math.pi*k/total + rot),
                                  top_z + thickness/2]
                                 for k in range(total)])
            ring_bot = ring_top.copy(); ring_bot[:,2] -= thickness/2

            # dessus
            Vt = np.vstack([[cx_s, cy, top_z+thickness/2], ring_top])
            traces.append(_mesh(Vt, _tri_fan(0, range(1,total+1)), star_color))
            # dessous
            Vb = np.vstack([[cx_s, cy, top_z-thickness/2], ring_bot])
            traces.append(_mesh(Vb, _tri_fan(0, range(total,0,-1)), star_color))
            # côtés
            for k in range(total):
                kk = (k+1)%total
                quad = np.array([ring_top[k], ring_top[kk],
                                 ring_bot[kk], ring_bot[k]])
                traces.append(_mesh(quad, [(0,1,2),(0,2,3)], star_color))

    return traces


# import plotly.graph_objects as go
# import plotly.io as pio

# fig = go.Figure()

# # hexagone noir, 3 étoiles
# for t in hex_prism_traces(r=0, q=0, fill_color='black', H=3, n_stars=3):
#     fig.add_trace(t)

# # hexagone bleu (r=0,q=1) sans étoiles
# for t in hex_prism_traces(r=0, q=1, fill_color='blue', H=6, n_stars=0):
#     fig.add_trace(t)

# # hexagone bleu (r=0,q=1) sans étoiles
# for t in hex_prism_traces(r=1, q=0, fill_color='purple', H=3, n_stars=1):
#     fig.add_trace(t)

# fig.update_layout(
#     scene=dict(
#         xaxis=dict(visible=False),
#         yaxis=dict(visible=False),
#         zaxis=dict(visible=False)
#     )
# )
# fig.update_layout(
#     scene=dict(
#         aspectmode="data",          # échelle identique X, Y, Z (facultatif)
#     ),
#     scene_camera=dict(
#         eye=dict(x=-2, y=-3, z=5),   # x,y petits → presque au-dessus
#         up=dict(x=0.1, y=0.8, z=0.5),          # garde le nord en haut
#         center=dict(x=0, y=0, z=0)       # point visé (souvent l’origine)
#     )
# )
# # fig.show()
# pio.write_image(fig, "plateau.png", scale=2)
