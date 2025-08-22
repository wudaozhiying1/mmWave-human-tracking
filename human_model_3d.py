#!/usr/bin/env python3
"""
3D Human Model Category
Used to display more realistic human models in PyQtGraph OpenGL.
"""

import numpy as np
import pyqtgraph.opengl as gl
from graph_utilities import getSphereMesh, getBoxLinesCoords

class HumanModel3D:
    
    def __init__(self, track_id, color=(1, 0, 0, 1)):
        self.track_id = track_id
        self.color = color
        
        self.head_radius = 0.12
        self.torso_width = 0.4
        self.torso_height = 0.6
        self.torso_depth = 0.2
        self.arm_length = 0.6
        self.arm_radius = 0.05
        self.leg_length = 0.8
        self.leg_radius = 0.08
        

        self.head = None
        self.torso = None
        self.left_arm = None
        self.right_arm = None
        self.left_leg = None
        self.right_leg = None
        

        self.x = 0
        self.y = 0
        self.z = 0
        self.vx = 0
        self.vy = 0
        self.vz = 0
        

        self.create_body_parts()
    
    def create_body_parts(self):

        head_mesh = getSphereMesh(
            xRadius=self.head_radius,
            yRadius=self.head_radius,
            zRadius=self.head_radius,
            stacks=8,
            sectors=8
        )
        self.head = gl.GLMeshItem(
            vertexes=head_mesh,
            color=self.color,
            drawEdges=False,
            drawFaces=True,
            smooth=True
        )
        

        torso_verts = self.create_box_vertices(
            -self.torso_width/2, -self.torso_depth/2, 0,
            self.torso_width/2, self.torso_depth/2, self.torso_height
        )
        torso_faces = self.create_box_faces()
        self.torso = gl.GLMeshItem(
            vertexes=torso_verts,
            faces=torso_faces,
            color=self.color,
            drawEdges=False,
            drawFaces=True
        )
        

        self.left_arm = self.create_cylinder(
            radius=self.arm_radius,
            height=self.arm_length,
            color=self.color
        )
        

        self.right_arm = self.create_cylinder(
            radius=self.arm_radius,
            height=self.arm_length,
            color=self.color
        )
        

        self.left_leg = self.create_cylinder(
            radius=self.leg_radius,
            height=self.leg_length,
            color=self.color
        )
        

        self.right_leg = self.create_cylinder(
            radius=self.leg_radius,
            height=self.leg_length,
            color=self.color
        )
    
    def create_box_vertices(self, x1, y1, z1, x2, y2, z2):

        vertices = np.array([
            [x1, y1, z1], [x2, y1, z1], [x1, y2, z1], [x2, y2, z1],
            [x1, y1, z2], [x2, y1, z2], [x1, y2, z2], [x2, y2, z2]
        ])
        return vertices
    
    def create_box_faces(self):

        faces = np.array([
            [0, 1, 2], [1, 3, 2],  
            [4, 6, 5], [5, 6, 7],  
            [0, 4, 1], [1, 4, 5],  
            [2, 3, 6], [3, 7, 6],  
            [0, 2, 4], [2, 6, 4],  
            [1, 5, 3], [3, 5, 7]   
        ])
        return faces
    
    def create_cylinder(self, radius, height, color, segments=8):

        theta = np.linspace(0, 2*np.pi, segments+1)[:-1]
        z_coords = np.array([0, height])
        
        vertices = []
        for z in z_coords:
            for t in theta:
                vertices.append([radius*np.cos(t), radius*np.sin(t), z])
        
        vertices = np.array(vertices)
        

        faces = []
        for i in range(segments):

            faces.append([i, (i+1)%segments, i+segments])
            faces.append([(i+1)%segments, (i+1)%segments+segments, i+segments])
        

        for i in range(1, segments-1):
            faces.append([0, i, i+1])  
            faces.append([segments, segments+i+1, segments+i])  
        
        faces = np.array(faces)
        
        return gl.GLMeshItem(
            vertexes=vertices,
            faces=faces,
            color=color,
            drawEdges=False,
            drawFaces=True
        )
    
    def update_position(self, x, y, z, vx=0, vy=0, vz=0):

        self.x = x
        self.y = y
        self.z = z
        self.vx = vx
        self.vy = vy
        self.vz = vz
        

        head_z = z + self.torso_height + self.head_radius
        torso_z = z + self.torso_height/2
        arm_z = z + self.torso_height * 0.8
        leg_z = z + self.leg_length/2

        self.head.resetTransform()
        self.head.translate(x, y, head_z)
        

        self.torso.resetTransform()
        self.torso.translate(x, y, torso_z)
        

        self.left_arm.resetTransform()
        self.left_arm.translate(x - self.torso_width/2 - self.arm_radius, y, arm_z)
        self.left_arm.rotate(90, 0, 0, 1)  
        

        self.right_arm.resetTransform()
        self.right_arm.translate(x + self.torso_width/2 + self.arm_radius, y, arm_z)
        self.right_arm.rotate(90, 0, 0, 1) 
        

        self.left_leg.resetTransform()
        self.left_leg.translate(x - self.torso_width/4, y, leg_z)
        

        self.right_leg.resetTransform()
        self.right_leg.translate(x + self.torso_width/4, y, leg_z)
    
    def add_to_widget(self, gl_widget):

        gl_widget.addItem(self.head)
        gl_widget.addItem(self.torso)
        gl_widget.addItem(self.left_arm)
        gl_widget.addItem(self.right_arm)
        gl_widget.addItem(self.left_leg)
        gl_widget.addItem(self.right_leg)
    
    def remove_from_widget(self, gl_widget):

        gl_widget.removeItem(self.head)
        gl_widget.removeItem(self.torso)
        gl_widget.removeItem(self.left_arm)
        gl_widget.removeItem(self.right_arm)
        gl_widget.removeItem(self.left_leg)
        gl_widget.removeItem(self.right_leg)
    
    def set_visible(self, visible):

        self.head.setVisible(visible)
        self.torso.setVisible(visible)
        self.left_arm.setVisible(visible)
        self.right_arm.setVisible(visible)
        self.left_leg.setVisible(visible)
        self.right_leg.setVisible(visible)


class SimpleHumanModel:

    
    def __init__(self, track_id, color=(1, 0, 0, 1)):
        self.track_id = track_id
        self.color = color
        

        self.height = 1.7
        self.shoulder_width = 0.5
        self.hip_width = 0.4
        

        self.body_lines = None
        self.head_sphere = None
        

        self.x = 0
        self.y = 0
        self.z = 0
        
        self.create_body()
    
    def create_body(self):

        head_mesh = getSphereMesh(
            xRadius=0.08,
            yRadius=0.08,
            zRadius=0.08,
            stacks=6,
            sectors=6
        )
        self.head_sphere = gl.GLMeshItem(
            vertexes=head_mesh,
            color=self.color,
            drawEdges=False,
            drawFaces=True
        )
        

        self.body_lines = gl.GLLinePlotItem(
            pos=np.array([]),
            color=self.color,
            width=3,
            antialias=True
        )
    
    def update_position(self, x, y, z):

        self.x = x
        self.y = y
        self.z = z
        

        head_z = z + self.height
        neck_z = z + self.height - 0.1
        shoulder_z = z + self.height - 0.2
        hip_z = z + self.height * 0.6
        foot_z = z
        

        body_points = np.array([

            [x, y, head_z],
            [x, y, neck_z],
            

            [x, y, neck_z],
            [x, y, hip_z],
            

            [x - self.shoulder_width/2, y, shoulder_z],
            [x + self.shoulder_width/2, y, shoulder_z],
            

            [x - self.shoulder_width/2, y, shoulder_z],
            [x - self.shoulder_width/2 - 0.3, y, shoulder_z - 0.2],
            

            [x + self.shoulder_width/2, y, shoulder_z],
            [x + self.shoulder_width/2 + 0.3, y, shoulder_z - 0.2],
            

            [x - self.hip_width/4, y, hip_z],
            [x - self.hip_width/4, y, foot_z],
            

            [x + self.hip_width/4, y, hip_z],
            [x + self.hip_width/4, y, foot_z],
            

            [x - self.hip_width/2, y, hip_z],
            [x + self.hip_width/2, y, hip_z]
        ])
        
        self.body_lines.setData(pos=body_points)
        

        self.head_sphere.resetTransform()
        self.head_sphere.translate(x, y, head_z)
    
    def add_to_widget(self, gl_widget):

        gl_widget.addItem(self.head_sphere)
        gl_widget.addItem(self.body_lines)
    
    def remove_from_widget(self, gl_widget):

        gl_widget.removeItem(self.head_sphere)
        gl_widget.removeItem(self.body_lines)
    
    def set_visible(self, visible):

        self.head_sphere.setVisible(visible)
        self.body_lines.setVisible(visible)


class HumanModelFactory:

    
    @staticmethod
    def create_model(model_type, track_id, color):

        if model_type == "detailed":
            return HumanModel3D(track_id, color)
        elif model_type == "simple":
            return SimpleHumanModel(track_id, color)
        else:

            return SimpleHumanModel(track_id, color) 