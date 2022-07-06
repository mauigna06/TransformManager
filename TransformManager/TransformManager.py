import logging
import os

import vtk
import qt

import slicer
from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin


numberOfIntrinsicTransformsMax = 8

_AXES = {
    'XYZ': 0, 'XYX': 1, 'XZY': 2,
    'XZX': 3, 'YZX': 4, 'YZY': 5,
    'YXZ': 6, 'YXY': 7, 'ZXY': 8,
    'ZXZ': 9, 'ZYX': 10, 'ZYZ': 11,}





import math
import numpy as np

#source https://github.com/matthew-brett/transforms3d/blob/52321496a0d98f1f698fd3ed81f680d740202553/transforms3d/_gohlketransforms.py#L1680

# epsilon for testing whether a number is close to zero
_EPS = np.finfo(float).eps * 4.0

# axis sequences for Euler angles
_NEXT_AXIS = [1, 2, 0, 1]

# map axes strings to/from tuples of inner axis, parity, repetition, frame
_AXES2TUPLE = {
    'sxyz': (0, 0, 0, 0), 'sxyx': (0, 0, 1, 0), 'sxzy': (0, 1, 0, 0),
    'sxzx': (0, 1, 1, 0), 'syzx': (1, 0, 0, 0), 'syzy': (1, 0, 1, 0),
    'syxz': (1, 1, 0, 0), 'syxy': (1, 1, 1, 0), 'szxy': (2, 0, 0, 0),
    'szxz': (2, 0, 1, 0), 'szyx': (2, 1, 0, 0), 'szyz': (2, 1, 1, 0),
    'rzyx': (0, 0, 0, 1), 'rxyx': (0, 0, 1, 1), 'ryzx': (0, 1, 0, 1),
    'rxzx': (0, 1, 1, 1), 'rxzy': (1, 0, 0, 1), 'ryzy': (1, 0, 1, 1),
    'rzxy': (1, 1, 0, 1), 'ryxy': (1, 1, 1, 1), 'ryxz': (2, 0, 0, 1),
    'rzxz': (2, 0, 1, 1), 'rxyz': (2, 1, 0, 1), 'rzyz': (2, 1, 1, 1)}

_TUPLE2AXES = dict((v, k) for k, v in _AXES2TUPLE.items())

def euler_from_matrix(matrix, axes='sxyz'):
    """Return Euler angles from rotation matrix for specified axis sequence.
    axes : One of 24 axis sequences as string or encoded tuple
    Note that many Euler angle triplets can describe one matrix.
    >>> R0 = euler_matrix(1, 2, 3, 'syxz')
    >>> al, be, ga = euler_from_matrix(R0, 'syxz')
    >>> R1 = euler_matrix(al, be, ga, 'syxz')
    >>> np.allclose(R0, R1)
    True

    angles = (math.pi/2,math.pi/4,-math.pi/6)
    mydict = {'syxz': (1, 1, 0, 0)}
    for axes in mydict.keys():
        R0 = euler_matrix(axes=axes, *angles)
        R1 = euler_matrix(axes=axes, *euler_from_matrix(R0, axes))
        print(euler_from_matrix(R0, axes))
        if not np.allclose(R0, R1): print(axes, "failed")
    """
    try:
        firstaxis, parity, repetition, frame = _AXES2TUPLE[axes.lower()]
    except (AttributeError, KeyError):
        _TUPLE2AXES[axes]  # validation
        firstaxis, parity, repetition, frame = axes
    #
    i = firstaxis
    j = _NEXT_AXIS[i+parity]
    k = _NEXT_AXIS[i-parity+1]
    #
    M = np.array(matrix, dtype=np.float64, copy=True)[:3, :3]
    #print(M)
    #scales = np.linalg.norm(M,axis=1)
    #M[:,0] = M[:,0]/scales[0]
    #M[:,1] = M[:,1]/scales[1]
    #M[:,2] = M[:,2]/scales[2]
    if repetition:
        sy = math.sqrt(M[i, j]*M[i, j] + M[i, k]*M[i, k])
        if sy > _EPS:
            ax = math.atan2( M[i, j],  M[i, k])
            ay = math.atan2( sy,       M[i, i])
            az = math.atan2( M[j, i], -M[k, i])
        else:
            ax = math.atan2(-M[j, k],  M[j, j])
            ay = math.atan2( sy,       M[i, i])
            az = 0.0
    else:
        cy = math.sqrt(M[i, i]*M[i, i] + M[j, i]*M[j, i])
        if cy > _EPS:
            ax = math.atan2( M[k, j],  M[k, k])
            ay = math.atan2(-M[k, i],  cy)
            az = math.atan2( M[j, i],  M[i, i])
        else:
            ax = math.atan2(-M[j, k],  M[j, j])
            ay = math.atan2(-M[k, i],  cy)
            az = 0.0
    #
    if parity:
        ax, ay, az = -ax, -ay, -az
    if frame:
        ax, az = az, ax
    return ax, ay, az




def rotation_from_matrix(matrix):
    """Return rotation angle and axis from rotation matrix.

    >>> angle = (random.random() - 0.5) * (2*math.pi)
    >>> direc = np.random.random(3) - 0.5
    >>> point = np.random.random(3) - 0.5
    >>> R0 = rotation_matrix(angle, direc, point)
    >>> angle, direc, point = rotation_from_matrix(R0)
    >>> R1 = rotation_matrix(angle, direc, point)
    >>> is_same_transform(R0, R1)
    True

    """
    R = np.array(matrix, dtype=np.float64, copy=False)
    R33 = R[:3, :3]
    # direction: unit eigenvector of R33 corresponding to eigenvalue of 1
    l, W = np.linalg.eig(R33.T)
    i = np.where(abs(np.real(l) - 1.0) < 1e-8)[0]
    if not len(i):
        raise ValueError("no unit eigenvector corresponding to eigenvalue 1")
    direction = np.real(W[:, i[-1]]).squeeze()
    # point: unit eigenvector of R33 corresponding to eigenvalue of 1
    l, Q = np.linalg.eig(R)
    i = np.where(abs(np.real(l) - 1.0) < 1e-8)[0]
    if not len(i):
        raise ValueError("no unit eigenvector corresponding to eigenvalue 1")
    point = np.real(Q[:, i[-1]]).squeeze()
    point /= point[3]
    # rotation angle depending on direction
    cosa = (np.trace(R33) - 1.0) / 2.0
    if abs(direction[2]) > 1e-8:
        sina = (R[1, 0] + (cosa - 1.0) * direction[0] * direction[1]) / direction[2]
    elif abs(direction[1]) > 1e-8:
        sina = (R[0, 2] + (cosa - 1.0) * direction[0] * direction[2]) / direction[1]
    else:
        sina = (R[2, 1] + (cosa - 1.0) * direction[1] * direction[2]) / direction[0]
    angle = math.atan2(sina, cosa)
    return angle, direction, point







class AngleTransformWidget(qt.QWidget):
  def __init__(self, parent=None):
    super().__init__(parent)
    #
    #self.angleLabel = qt.QLabel("Deg")
    #
    self.degAngleSpinbox = qt.QDoubleSpinBox()
    self.degAngleSpinbox.value = 0
    self.degAngleSpinbox.maximum = 1e6
    self.degAngleSpinbox.minimum = -1e6
    #
    #self.axisLabel = qt.QLabel("Axis")
    #
    self.axisComboBox = qt.QComboBox()
    self.axisComboBox.addItem("X")
    self.axisComboBox.addItem("Y")
    self.axisComboBox.addItem("Z")
    #
    self.definedLayout = qt.QHBoxLayout()
    #self.definedLayout.addWidget(self.angleLabel)
    self.definedLayout.addWidget(self.degAngleSpinbox)
    #self.definedLayout.addWidget(self.axisLabel)
    self.definedLayout.addWidget(self.axisComboBox)
    #
    self.setLayout(self.definedLayout)


#
# TransformManager
#

class TransformManager(ScriptedLoadableModule):
  """Uses ScriptedLoadableModule base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

  def __init__(self, parent):
    ScriptedLoadableModule.__init__(self, parent)
    self.parent.title = "TransformManager"  # TODO: make this more human readable by adding spaces
    self.parent.categories = ["Examples"]  # TODO: set categories (folders where the module shows up in the module selector)
    self.parent.dependencies = []  # TODO: add here list of module names that this module requires
    self.parent.contributors = ["John Doe (AnyWare Corp.)"]  # TODO: replace with "Firstname Lastname (Organization)"
    # TODO: update with short description of the module and a link to online module documentation
    self.parent.helpText = """
This is an example of scripted loadable module bundled in an extension.
See more information in <a href="https://github.com/organization/projectname#TransformManager">module documentation</a>.
"""
    # TODO: replace with organization, grant and thanks
    self.parent.acknowledgementText = """
This file was originally developed by Jean-Christophe Fillion-Robin, Kitware Inc., Andras Lasso, PerkLab,
and Steve Pieper, Isomics, Inc. and was partially funded by NIH grant 3P41RR013218-12S1.
"""

#
# TransformManagerWidget
#

class TransformManagerWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
  """Uses ScriptedLoadableModuleWidget base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

  def __init__(self, parent=None):
    """
    Called when the user opens the module the first time and the widget is initialized.
    """
    ScriptedLoadableModuleWidget.__init__(self, parent)
    VTKObservationMixin.__init__(self)  # needed for parameter node observation
    self.logic = None
    self._parameterNode = None
    self._updatingGUIFromParameterNode = False
    self._transformObserver = 0
    self._updatingGUIFromTransformNode = False

  def setup(self):
    """
    Called when the user opens the module the first time and the widget is initialized.
    """
    ScriptedLoadableModuleWidget.setup(self)

    # Load widget from .ui file (created by Qt Designer).
    # Additional widgets can be instantiated manually and added to self.layout.
    uiWidget = slicer.util.loadUI(self.resourcePath('UI/TransformManager.ui'))
    self.layout.addWidget(uiWidget)
    self.ui = slicer.util.childWidgetVariables(uiWidget)

    self.ui.nIntrinsicTransformsFrameLayout = qt.QVBoxLayout()
    for i in range(numberOfIntrinsicTransformsMax):
      intrinsicAngleTransformWidget = AngleTransformWidget()
      intrinsicAngleTransformWidget.objectName = "intrinsicAngleTransformWidget" + str(i)
      intrinsicAngleTransformWidget.visible = False
      intrinsicAngleTransformWidget.enabled = False
      intrinsicAngleTransformWidget.axisComboBox.currentIndexChanged.connect(self.updateParameterNodeFromGUI)
      intrinsicAngleTransformWidget.degAngleSpinbox.valueChanged.connect(self.updateParameterNodeFromGUI)
      self.ui.nIntrinsicTransformsFrameLayout.addWidget(intrinsicAngleTransformWidget)

    self.ui.nIntrinsicTransformsFrame.setLayout(self.ui.nIntrinsicTransformsFrameLayout)

    self.ui.xPivotPointInfinitLabel.hide()
    self.ui.yPivotPointInfinitLabel.hide()
    self.ui.zPivotPointInfinitLabel.hide()
    self.ui.infinitPivotPointLabel.hide()


    for key in _AXES.keys():
      self.ui.premultiplyAxesComboBox.addItem(key)

    # Set scene in MRML widgets. Make sure that in Qt designer the top-level qMRMLWidget's
    # "mrmlSceneChanged(vtkMRMLScene*)" signal in is connected to each MRML widget's.
    # "setMRMLScene(vtkMRMLScene*)" slot.
    uiWidget.setMRMLScene(slicer.mrmlScene)

    # Create logic class. Logic implements all computations that should be possible to run
    # in batch mode, without a graphical user interface.
    self.logic = TransformManagerLogic()

    # Connections

    # These connections ensure that we update parameter node when scene is closed
    self.addObserver(slicer.mrmlScene, slicer.mrmlScene.StartCloseEvent, self.onSceneStartClose)
    self.addObserver(slicer.mrmlScene, slicer.mrmlScene.EndCloseEvent, self.onSceneEndClose)

    # These connections ensure that whenever user changes some settings on the GUI, that is saved in the MRML scene
    # (in the selected parameter node).
    #self.ui.
    self.ui.premultiplyAxesComboBox.currentTextChanged.connect(self.updateParameterNodeFromGUI)
    self.ui.transformNodeComboBox.connect("currentNodeChanged(vtkMRMLNode*)", self.updateParameterNodeFromGUI)
    self.ui.transformNodeComboBox.connect("currentNodeChanged(vtkMRMLNode*)", self.setTransformObserver)
    self.ui.outputPlaneNodeComboBox.connect("currentNodeChanged(vtkMRMLNode*)", self.onOutputPlaneNodeChanged)
    self.ui.outputAngleNodeComboBox.connect("currentNodeChanged(vtkMRMLNode*)", self.onOutputAngleNodeChanged)
    #self.ui.outputSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.updateParameterNodeFromGUI)
    #self.ui.imageThresholdSliderWidget.connect("valueChanged(double)", self.updateParameterNodeFromGUI)
    #self.ui.invertOutputCheckBox.connect("toggled(bool)", self.updateParameterNodeFromGUI)
    #self.ui.invertedOutputSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.updateParameterNodeFromGUI)

    # Buttons
    self.ui.appendIntrinsicTransformButton.connect('clicked(bool)', self.onAppendIntrinsicTransformButton)
    self.ui.deleteLastIntrinsicTransformButton.connect('clicked(bool)', self.onDeleteLastIntrinsicTransformButton)

    self.ui.xRotationAxisSpinBox.valueChanged.connect(self.updateNormalizedAxis)
    self.ui.yRotationAxisSpinBox.valueChanged.connect(self.updateNormalizedAxis)
    self.ui.zRotationAxisSpinBox.valueChanged.connect(self.updateNormalizedAxis)
    #
    self.ui.angleOfRotationSpinBox.valueChanged.connect(self.updateTransformFromWidgets2)
    #self.ui.xNormalizedRotationAxisSpinBox.valueChanged.connect(self.updateTransformFromWidgets2)
    #self.ui.yNormalizedRotationAxisSpinBox.valueChanged.connect(self.updateTransformFromWidgets2)
    #self.ui.zNormalizedRotationAxisSpinBox.valueChanged.connect(self.updateTransformFromWidgets2)
    self.ui.xPivotPointSpinBox.valueChanged.connect(self.updateTransformFromWidgets2)
    self.ui.yPivotPointSpinBox.valueChanged.connect(self.updateTransformFromWidgets2)
    self.ui.zPivotPointSpinBox.valueChanged.connect(self.updateTransformFromWidgets2)

    self.ui.translationTransformSliders.valuesChanged.connect(self.updateParameterNodeFromGUI)
    #self.ui.intrinsicTransformMatrixWidget.valueChanged.connect(self.updateTransformNode)


    # Make sure parameter node is initialized (needed for module reload)
    self.initializeParameterNode()

  def onOutputPlaneNodeChanged(self,caller=None,event=None):
    if caller is not None:
      self._parameterNode.SetNodeReferenceID("outputPlane", caller.GetID())
      #
      planeNode = caller
      planeNode.SetPlaneType(planeNode.PlaneType3Points)
      planeNode.SetOrigin([0,0,0])
      planeNode.SetAxes(
          [1,0,0],
          [0,1,0],
          [0,0,1]
      )
      planeNode.SetNthControlPointVisibility(1,False)
      planeNode.SetNthControlPointVisibility(2,False)
      planeNode.SetSizeMode(planeNode.SizeModeAbsolute)
      planeNode.SetSize(50,50)
      displayNode = planeNode.GetDisplayNode()
      displayNode.SetScaleHandleVisibility(False)
      displayNode.SetTranslationHandleVisibility(True)
      displayNode.SetTranslationHandleComponentVisibility(
          True,
          True,
          True,
          False
      )
  
  def onOutputAngleNodeChanged(self,caller=None,event=None):
    if caller is not None:
      self._parameterNode.SetNodeReferenceID("outputAngle", caller.GetID())
  
  def updateNormalizedAxis(self):
    #print("updateNormalizedAxis")
    axisDirection = np.array([
      self.ui.xRotationAxisSpinBox.value,
      self.ui.yRotationAxisSpinBox.value,
      self.ui.zRotationAxisSpinBox.value,
    ])
    axisNormalized = axisDirection/np.linalg.norm(axisDirection)
    signalBlocked = self.ui.xNormalizedRotationAxisSpinBox.blockSignals(True)
    self.ui.xNormalizedRotationAxisSpinBox.value = axisNormalized[0]
    self.ui.xNormalizedRotationAxisSpinBox.blockSignals(signalBlocked)
    signalBlocked = self.ui.yNormalizedRotationAxisSpinBox.blockSignals(True)
    self.ui.yNormalizedRotationAxisSpinBox.value = axisNormalized[1]
    self.ui.yNormalizedRotationAxisSpinBox.blockSignals(signalBlocked)
    signalBlocked = self.ui.zNormalizedRotationAxisSpinBox.blockSignals(True)
    self.ui.zNormalizedRotationAxisSpinBox.value = axisNormalized[2]
    self.ui.zNormalizedRotationAxisSpinBox.blockSignals(signalBlocked)
    print(
      self.ui.xNormalizedRotationAxisSpinBox.value,
      self.ui.yNormalizedRotationAxisSpinBox.value,
      self.ui.zNormalizedRotationAxisSpinBox.value
    )
    self.updateTransformFromWidgets2()
  
  def setTransformObserver(self,caller=None,event=None):
    intrinsicTransformNode = self._parameterNode.GetNodeReference("intrinsicTransformNode")
    if intrinsicTransformNode:
      intrinsicTransformNode.RemoveObserver(self._transformObserver)
    if caller is not None:
      self._transformObserver = caller.AddObserver(
        slicer.vtkMRMLTransformNode.TransformModifiedEvent, self.updateTransformationWidgetsFromNode
      )

  def updateTransformationWidgetsFromNode(self,caller=None,event=None):
    intrinsicTransformNode = self._parameterNode.GetNodeReference("intrinsicTransformNode")
    
    if intrinsicTransformNode is None or self._updatingGUIFromTransformNode:
      return

    self._updatingGUIFromTransformNode = True

    print("updateTransformationWidgetsFromNode")

    #self.ui.intrinsicAnglesLabel
    axis = self._parameterNode.GetParameter("decodeIntrinsicAnglesOrder")
    anglesRad = euler_from_matrix(
      slicer.util.arrayFromVTKMatrix(
        intrinsicTransformNode.GetMatrixTransformToParent()
      ),
      's'+axis[::-1]
    )

    anglesDeg = [
      vtk.vtkMath.DegreesFromRadians(anglesRad[2]),
      vtk.vtkMath.DegreesFromRadians(anglesRad[1]),
      vtk.vtkMath.DegreesFromRadians(anglesRad[0]),
    ]

    labelText = ""
    for i in range(3):
      labelText += axis[i]+"="+str(anglesDeg[i])+","
    labelText = labelText[:-1]

    #print(f"set text to {labelText}")
    self.ui.intrinsicAnglesLabel.setText(labelText)

    transform = vtk.vtkTransform()
    transform.PostMultiply()
    transform.Concatenate(
      intrinsicTransformNode.GetMatrixTransformToParent()
    )

    tMatrix = transform.GetMatrix()
    self.ui.intrinsicTransformMatrixWidget.editable = True
    signalBlocked = self.ui.intrinsicTransformMatrixWidget.blockSignals(True)
    for i in range(3):
      for j in range(3):
        self.ui.intrinsicTransformMatrixWidget.setValue(i,j,tMatrix.GetElement(i,j))
    self.ui.intrinsicTransformMatrixWidget.blockSignals(signalBlocked)
    self.ui.intrinsicTransformMatrixWidget.editable = False

    angleAndRotationAxis = [0,0,0,0]
    transform.GetOrientationWXYZ(angleAndRotationAxis)
    #angle
    signalBlocked = self.ui.angleOfRotationSpinBox.blockSignals(True)
    self.ui.angleOfRotationSpinBox.value = angleAndRotationAxis[0]
    self.ui.angleOfRotationSpinBox.blockSignals(signalBlocked)
    #axisx
    signalBlocked = self.ui.xNormalizedRotationAxisSpinBox.blockSignals(True)
    self.ui.xNormalizedRotationAxisSpinBox.value = angleAndRotationAxis[1]
    self.ui.xNormalizedRotationAxisSpinBox.blockSignals(signalBlocked)
    #axisy
    signalBlocked = self.ui.yNormalizedRotationAxisSpinBox.blockSignals(True)
    self.ui.yNormalizedRotationAxisSpinBox.value = angleAndRotationAxis[2]
    self.ui.yNormalizedRotationAxisSpinBox.blockSignals(signalBlocked)
    #axisz
    signalBlocked = self.ui.zNormalizedRotationAxisSpinBox.blockSignals(True)
    self.ui.zNormalizedRotationAxisSpinBox.value = angleAndRotationAxis[3]
    self.ui.zNormalizedRotationAxisSpinBox.blockSignals(signalBlocked)
    #
    #
    angle, direction, pivotPoint = rotation_from_matrix(
      slicer.util.arrayFromVTKMatrix(tMatrix)
    )
    #
    origin = [0,0,0]
    transform.TransformPoint(origin,origin)
    #
    if np.linalg.norm(np.array(pivotPoint)) > 1e6:
      self.ui.xCurrentPivotPointSpinBox.hide()
      self.ui.yCurrentPivotPointSpinBox.hide()
      self.ui.zCurrentPivotPointSpinBox.hide()
      self.ui.currentPivotPointLabel.hide()
      self.ui.xPivotPointInfinitLabel.show()
      self.ui.yPivotPointInfinitLabel.show()
      self.ui.zPivotPointInfinitLabel.show()
      self.ui.infinitPivotPointLabel.show()
      self.ui.outputAngleNodeComboBox.enabled = False
    else:
      #solve pivotPointNearestToOrigin
      scalar = (
        -(
        angleAndRotationAxis[1]*pivotPoint[0] +
        angleAndRotationAxis[2]*pivotPoint[1] +
        angleAndRotationAxis[3]*pivotPoint[2]
        )/(
          (angleAndRotationAxis[1])**2 +
          (angleAndRotationAxis[2])**2 +
          (angleAndRotationAxis[3])**2
        )
      )
      nearestPoint = [
        angleAndRotationAxis[1]*scalar+pivotPoint[0],
        angleAndRotationAxis[2]*scalar+pivotPoint[1],
        angleAndRotationAxis[3]*scalar+pivotPoint[2]
      ]
      #pivotPointx
      signalBlocked = self.ui.xCurrentPivotPointSpinBox.blockSignals(True)
      self.ui.xCurrentPivotPointSpinBox.value = nearestPoint[0]
      self.ui.xCurrentPivotPointSpinBox.blockSignals(signalBlocked)
      #pivotPointy
      signalBlocked = self.ui.yCurrentPivotPointSpinBox.blockSignals(True)
      self.ui.yCurrentPivotPointSpinBox.value = nearestPoint[1]
      self.ui.yCurrentPivotPointSpinBox.blockSignals(signalBlocked)
      #pivotPointz
      signalBlocked = self.ui.zCurrentPivotPointSpinBox.blockSignals(True)
      self.ui.zCurrentPivotPointSpinBox.value = nearestPoint[2]
      self.ui.zCurrentPivotPointSpinBox.blockSignals(signalBlocked)
      #showhide
      self.ui.xCurrentPivotPointSpinBox.show()
      self.ui.yCurrentPivotPointSpinBox.show()
      self.ui.zCurrentPivotPointSpinBox.show()
      self.ui.currentPivotPointLabel.show()
      self.ui.xPivotPointInfinitLabel.hide()
      self.ui.yPivotPointInfinitLabel.hide()
      self.ui.zPivotPointInfinitLabel.hide()
      self.ui.infinitPivotPointLabel.hide()
      self.ui.outputAngleNodeComboBox.enabled = True

      outputAngle = self._parameterNode.GetNodeReference("outputAngle")
      if outputAngle:
        if outputAngle.GetNthControlPointPosition(0) is None:
          outputAngle.AddControlPoint([0,0,0])
        else:
          outputAngle.SetNthControlPointPositionFromArray(0,[0,0,0])
        #
        if outputAngle.GetNthControlPointPosition(1) is None:
          outputAngle.AddControlPoint(nearestPoint)
        else:
          outputAngle.SetNthControlPointPositionFromArray(1,nearestPoint)
        #
        if outputAngle.GetNthControlPointPosition(2) is None:
          outputAngle.AddControlPoint(origin)
        else:
          outputAngle.SetNthControlPointPositionFromArray(2,origin)
        #
        outputAngle.SetLocked(True)

    outputPlane = self._parameterNode.GetNodeReference("outputPlane")
    if outputPlane:
      planeTransform = vtk.vtkTransform()
      planeTransform.PostMultiply()
      planeTransform.Concatenate(
        intrinsicTransformNode.GetMatrixTransformToParent()
      )
      #transformPoints
      point0 = [0,0,0]
      point1 = [0,0,0]
      point2 = [0,0,0]
      outputPlane.GetNthControlPointPosition(0,point0)
      outputPlane.GetNthControlPointPosition(1,point1)
      outputPlane.GetNthControlPointPosition(2,point2)
      #
      planeTransform.TransformPoint(point0,point0)
      planeTransform.TransformPoint(point1,point1)
      planeTransform.TransformPoint(point2,point2)
      #
      outputPlane.SetNthControlPointPositionFromArray(0,point0)
      outputPlane.SetNthControlPointPositionFromArray(1,point1)
      outputPlane.SetNthControlPointPositionFromArray(2,point2)
      #
      outputPlane.SetLocked(True)

    self._updatingGUIFromTransformNode = False

  def updateTransformFromWidgets2(self):
    intrinsicTransformNode = self._parameterNode.GetNodeReference("intrinsicTransformNode")
    
    if intrinsicTransformNode is None or self._updatingGUIFromTransformNode:
      return
    
    print("updateTransformFromWidgets2")

    rotationTransform = vtk.vtkTransform()
    rotationTransform.PostMultiply()
    rotationTransform.Translate(
      -self.ui.xPivotPointSpinBox.value,
      -self.ui.yPivotPointSpinBox.value,
      -self.ui.zPivotPointSpinBox.value
    )
    rotationTransform.RotateWXYZ(
      self.ui.angleOfRotationSpinBox.value,
      self.ui.xNormalizedRotationAxisSpinBox.value,
      self.ui.yNormalizedRotationAxisSpinBox.value,
      self.ui.zNormalizedRotationAxisSpinBox.value,
    )
    rotationTransform.Translate(
      self.ui.xPivotPointSpinBox.value,
      self.ui.yPivotPointSpinBox.value,
      self.ui.zPivotPointSpinBox.value
    )

    #currentMatrix = intrinsicTransformNode.GetMatrixTransformToParent()
    #currentTranslationTransform = vtk.vtkTransform()
    #currentTranslationTransform.PostMultiply()
    #inverseRotationTransform


    #transform.Translate(
    #  currentMatrix.GetElement(0,3),
    #  currentMatrix.GetElement(1,3),
    #  currentMatrix.GetElement(2,3)
    #)
    #intrinsicTransformNode.RemoveObserver(self._transformObserver)
    intrinsicTransformNode.SetMatrixTransformToParent(rotationTransform.GetMatrix())
    #self.setTransformObserver(intrinsicTransformNode)

  def updateTransformFromWidgets1(self):
    intrinsicTransformNode = self._parameterNode.GetNodeReference("intrinsicTransformNode")
    
    if intrinsicTransformNode is None or self._updatingGUIFromTransformNode:
      return

    print("updateTransformFromWidgets1")

    axisPreMultiplyString = self._parameterNode.GetParameter("axisPreMultiplyString")
    numberOfIntrinsicTransforms = int(self._parameterNode.GetParameter("numberOfIntrinsicTransforms"))

    transform = vtk.vtkTransform()
    transform.PostMultiply()
    for i in range(numberOfIntrinsicTransforms):
      currentAxis = axisPreMultiplyString[i]
      currentAngle = float(self._parameterNode.GetParameter("angle_"+str(i)))
      axisOfRotation = [int(currentAxis=="X"),int(currentAxis=="Y"),int(currentAxis=="Z"),0]
      transform.GetMatrix().MultiplyPoint(axisOfRotation,axisOfRotation)
      transform.RotateWXYZ(currentAngle,axisOfRotation[0],axisOfRotation[1],axisOfRotation[2])

    #print(transform.GetMatrix())

    rMatrix = transform.GetMatrix()
    currentMatrix = intrinsicTransformNode.GetMatrixTransformToParent()
    
    for i in range(3):
      for j in range(3):
        currentMatrix.SetElement(i,j,rMatrix.GetElement(i,j))

    intrinsicTransformNode.SetMatrixTransformToParent(currentMatrix)

  def onAppendIntrinsicTransformButton(self):
    if self._parameterNode is None or self._updatingGUIFromParameterNode:
      return

    wasModified = self._parameterNode.StartModify()  # Modify all properties in a single batch

    numberOfIntrinsicTransforms = int(self._parameterNode.GetParameter("numberOfIntrinsicTransforms"))
    if numberOfIntrinsicTransforms < (numberOfIntrinsicTransformsMax -1):
      numberOfIntrinsicTransforms += 1
    self._parameterNode.SetParameter("numberOfIntrinsicTransforms",str(numberOfIntrinsicTransforms))

    self._parameterNode.EndModify(wasModified)

  def onDeleteLastIntrinsicTransformButton(self):
    if self._parameterNode is None or self._updatingGUIFromParameterNode:
      return

    wasModified = self._parameterNode.StartModify()  # Modify all properties in a single batch

    numberOfIntrinsicTransforms = int(self._parameterNode.GetParameter("numberOfIntrinsicTransforms"))
    if numberOfIntrinsicTransforms > 1:
      numberOfIntrinsicTransforms -= 1
    self._parameterNode.SetParameter("numberOfIntrinsicTransforms",str(numberOfIntrinsicTransforms))
    self._parameterNode.SetParameter("angle_"+str(numberOfIntrinsicTransforms),str(0))
    axisPreMultiplyString = self._parameterNode.GetParameter("axisPreMultiplyString")
    self._parameterNode.SetParameter(
      "axisPreMultiplyString",
      axisPreMultiplyString[:numberOfIntrinsicTransforms] + "X" + axisPreMultiplyString[numberOfIntrinsicTransforms+1:]
    )

    self._parameterNode.EndModify(wasModified)

  def cleanup(self):
    """
    Called when the application closes and the module widget is destroyed.
    """
    self.removeObservers()

  def enter(self):
    """
    Called each time the user opens this module.
    """
    # Make sure parameter node exists and observed
    self.initializeParameterNode()

  def exit(self):
    """
    Called each time the user opens a different module.
    """
    # Do not react to parameter node changes (GUI wlil be updated when the user enters into the module)
    self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self.updateGUIFromParameterNode)

  def onSceneStartClose(self, caller, event):
    """
    Called just before the scene is closed.
    """
    # Parameter node will be reset, do not use it anymore
    self.setParameterNode(None)

  def onSceneEndClose(self, caller, event):
    """
    Called just after the scene is closed.
    """
    # If this module is shown while the scene is closed then recreate a new parameter node immediately
    if self.parent.isEntered:
      self.initializeParameterNode()

  def initializeParameterNode(self):
    """
    Ensure parameter node exists and observed.
    """
    # Parameter node stores all user choices in parameter values, node selections, etc.
    # so that when the scene is saved and reloaded, these settings are restored.

    self.setParameterNode(self.logic.getParameterNode())

    # Select default input nodes if nothing is selected yet to save a few clicks for the user
    if not self._parameterNode.GetNodeReference("InputVolume"):
      firstVolumeNode = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLScalarVolumeNode")
      if firstVolumeNode:
        self._parameterNode.SetNodeReferenceID("InputVolume", firstVolumeNode.GetID())

  def setParameterNode(self, inputParameterNode):
    """
    Set and observe parameter node.
    Observation is needed because when the parameter node is changed then the GUI must be updated immediately.
    """

    if inputParameterNode:
      self.logic.setDefaultParameters(inputParameterNode)

    # Unobserve previously selected parameter node and add an observer to the newly selected.
    # Changes of parameter node are observed so that whenever parameters are changed by a script or any other module
    # those are reflected immediately in the GUI.
    if self._parameterNode is not None:
      self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self.updateGUIFromParameterNode)
    self._parameterNode = inputParameterNode
    if self._parameterNode is not None:
      self.addObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self.updateGUIFromParameterNode)

    # Initial GUI update
    self.updateGUIFromParameterNode()

  def updateGUIFromParameterNode(self, caller=None, event=None):
    """
    This method is called whenever parameter node is changed.
    The module GUI is updated to show the current state of the parameter node.
    """

    if self._parameterNode is None or self._updatingGUIFromParameterNode:
      return

    # Make sure GUI changes do not call updateParameterNodeFromGUI (it could cause infinite loop)
    self._updatingGUIFromParameterNode = True

    #print("updatingGUI")

    axisPreMultiplyString = self._parameterNode.GetParameter("axisPreMultiplyString")
    numberOfIntrinsicTransforms = int(self._parameterNode.GetParameter("numberOfIntrinsicTransforms"))
    for i in range(numberOfIntrinsicTransformsMax):
      intrinsicAngleTransformWidget = self.ui.nIntrinsicTransformsFrame.findChild(
        "QWidget",
        "intrinsicAngleTransformWidget"+str(i)
      )
      intrinsicAngleTransformWidget.axisComboBox.currentText = axisPreMultiplyString[i]
      intrinsicAngleTransformWidget.degAngleSpinbox.value = float(self._parameterNode.GetParameter("angle_"+str(i)))
      if i < numberOfIntrinsicTransforms:
        intrinsicAngleTransformWidget.show()
      else:
        intrinsicAngleTransformWidget.hide()

    # Update node selectors and sliders
    self.ui.premultiplyAxesComboBox.setCurrentText(self._parameterNode.GetParameter("decodeIntrinsicAnglesOrder"))
    self.ui.transformNodeComboBox.setCurrentNode(self._parameterNode.GetNodeReference("intrinsicTransformNode"))
    
    #self.ui.outputSelector.setCurrentNode(self._parameterNode.GetNodeReference("OutputVolume"))
    #self.ui.invertedOutputSelector.setCurrentNode(self._parameterNode.GetNodeReference("OutputVolumeInverse"))
    #self.ui.imageThresholdSliderWidget.value = float(self._parameterNode.GetParameter("Threshold"))
    #self.ui.invertOutputCheckBox.checked = (self._parameterNode.GetParameter("Invert") == "true")

    # All the GUI updates are done
    self._updatingGUIFromParameterNode = False

  def updateParameterNodeFromGUI(self, caller=None, event=None):
    """
    This method is called when the user makes any change in the GUI.
    The changes are saved into the parameter node (so that they are restored when the scene is saved and loaded).
    """

    if self._parameterNode is None or self._updatingGUIFromParameterNode:
      return

    wasModified = self._parameterNode.StartModify()  # Modify all properties in a single batch

    #print("updatingParameterNode")

    equalIntrinsicTransformationParameters = True
    previousAxisPreMultiplyString = self._parameterNode.GetParameter("axisPreMultiplyString")
    previousAnglesList = []
    for i in range(numberOfIntrinsicTransformsMax):
      previousAnglesList.append(self._parameterNode.GetParameter("angle_"+str(i)))

    axisPreMultiplyString = ""
    numberOfIntrinsicTransforms = int(self._parameterNode.GetParameter("numberOfIntrinsicTransforms"))
    for i in range(numberOfIntrinsicTransformsMax):
      intrinsicAngleTransformWidget = self.ui.nIntrinsicTransformsFrame.findChild(
        "QWidget",
        "intrinsicAngleTransformWidget"+str(i)
      )
      #
      if i < numberOfIntrinsicTransforms:
        axisPreMultiplyString += intrinsicAngleTransformWidget.axisComboBox.currentText
        angleValue = intrinsicAngleTransformWidget.degAngleSpinbox.value
      else:
        axisPreMultiplyString += "X"
        angleValue = 0
      self._parameterNode.SetParameter("angle_"+str(i),str(angleValue))
      if equalIntrinsicTransformationParameters:
        equalIntrinsicTransformationParameters = np.isclose(angleValue,float(previousAnglesList[i]))
    #
    self._parameterNode.SetParameter("axisPreMultiplyString",axisPreMultiplyString)
    if equalIntrinsicTransformationParameters:
      equalIntrinsicTransformationParameters = previousAxisPreMultiplyString == axisPreMultiplyString
    #
    self._parameterNode.SetNodeReferenceID("intrinsicTransformNode", self.ui.transformNodeComboBox.currentNodeID)
    if self.ui.transformNodeComboBox.currentNodeID:
      self.ui.intrinsicTransformMatrixWidget.setMRMLTransformNode(
        slicer.mrmlScene.GetNodeByID(
          self.ui.transformNodeComboBox.currentNodeID
        )
      )
      self.ui.translationTransformSliders.setMRMLTransformNode(
        slicer.mrmlScene.GetNodeByID(
          self.ui.transformNodeComboBox.currentNodeID
        )
      )
    #
    if self.ui.transformNodeComboBox.currentNodeID != "":
      widgetsEnabled = True
    else:
      widgetsEnabled = False
    self.ui.angleOfRotationSpinBox.enabled = widgetsEnabled
    self.ui.xNormalizedRotationAxisSpinBox.enabled = False
    self.ui.yNormalizedRotationAxisSpinBox.enabled = False
    self.ui.zNormalizedRotationAxisSpinBox.enabled = False
    self.ui.xRotationAxisSpinBox.enabled = widgetsEnabled
    self.ui.yRotationAxisSpinBox.enabled = widgetsEnabled
    self.ui.zRotationAxisSpinBox.enabled = widgetsEnabled
    self.ui.xPivotPointSpinBox.enabled = widgetsEnabled
    self.ui.yPivotPointSpinBox.enabled = widgetsEnabled
    self.ui.zPivotPointSpinBox.enabled = widgetsEnabled
    self.ui.xCurrentPivotPointSpinBox.enabled = False
    self.ui.yCurrentPivotPointSpinBox.enabled = False
    self.ui.zCurrentPivotPointSpinBox.enabled = False
    #
    for i in range(numberOfIntrinsicTransformsMax):
      intrinsicAngleTransformWidget = self.ui.nIntrinsicTransformsFrame.findChild(
        "QWidget",
        "intrinsicAngleTransformWidget"+str(i)
      )
      #
      intrinsicAngleTransformWidget.enabled = widgetsEnabled

    self._parameterNode.SetParameter("decodeIntrinsicAnglesOrder",self.ui.premultiplyAxesComboBox.currentText)

    #self._parameterNode.SetNodeReferenceID("OutputVolume", self.ui.outputSelector.currentNodeID)
    #self._parameterNode.SetParameter("Threshold", str(self.ui.imageThresholdSliderWidget.value))
    #self._parameterNode.SetParameter("Invert", "true" if self.ui.invertOutputCheckBox.checked else "false")
    #self._parameterNode.SetNodeReferenceID("OutputVolumeInverse", self.ui.invertedOutputSelector.currentNodeID)

    self._parameterNode.EndModify(wasModified)

    if self.ui.transformNodeComboBox.currentNodeID != "":
      if not equalIntrinsicTransformationParameters:
        self.updateTransformFromWidgets1()

  def onApplyButton(self):
    """
    Run processing when user clicks "Apply" button.
    """
    with slicer.util.tryWithErrorDisplay("Failed to compute results.", waitCursor=True):

      # Compute output
      self.logic.process(self.ui.inputSelector.currentNode(), self.ui.outputSelector.currentNode(),
        self.ui.imageThresholdSliderWidget.value, self.ui.invertOutputCheckBox.checked)

      # Compute inverted output (if needed)
      if self.ui.invertedOutputSelector.currentNode():
        # If additional output volume is selected then result with inverted threshold is written there
        self.logic.process(self.ui.inputSelector.currentNode(), self.ui.invertedOutputSelector.currentNode(),
          self.ui.imageThresholdSliderWidget.value, not self.ui.invertOutputCheckBox.checked, showResult=False)


#
# TransformManagerLogic
#

class TransformManagerLogic(ScriptedLoadableModuleLogic):
  """This class should implement all the actual
  computation done by your module.  The interface
  should be such that other python code can import
  this class and make use of the functionality without
  requiring an instance of the Widget.
  Uses ScriptedLoadableModuleLogic base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

  def __init__(self):
    """
    Called when the logic class is instantiated. Can be used for initializing member variables.
    """
    ScriptedLoadableModuleLogic.__init__(self)

  def setDefaultParameters(self, parameterNode):
    """
    Initialize parameter node with default settings.
    """
    if not parameterNode.GetParameter("numberOfIntrinsicTransforms"):
      parameterNode.SetParameter("numberOfIntrinsicTransforms", "1")
    axisPreMultiplyString = ""
    for i in range(numberOfIntrinsicTransformsMax):
      if not parameterNode.GetParameter("angle_"+str(i)):
        parameterNode.SetParameter("angle_"+str(i), "0")
        axisPreMultiplyString += "X"
    if not parameterNode.GetParameter("axisPreMultiplyString"):
        parameterNode.SetParameter("axisPreMultiplyString", axisPreMultiplyString)
    if not parameterNode.GetParameter("decodeIntrinsicAnglesOrder"):
        parameterNode.SetParameter("decodeIntrinsicAnglesOrder", "XYZ")

  def process(self, inputVolume, outputVolume, imageThreshold, invert=False, showResult=True):
    """
    Run the processing algorithm.
    Can be used without GUI widget.
    :param inputVolume: volume to be thresholded
    :param outputVolume: thresholding result
    :param imageThreshold: values above/below this threshold will be set to 0
    :param invert: if True then values above the threshold will be set to 0, otherwise values below are set to 0
    :param showResult: show output volume in slice viewers
    """

    if not inputVolume or not outputVolume:
      raise ValueError("Input or output volume is invalid")

    import time
    startTime = time.time()
    logging.info('Processing started')

    # Compute the thresholded output volume using the "Threshold Scalar Volume" CLI module
    cliParams = {
      'InputVolume': inputVolume.GetID(),
      'OutputVolume': outputVolume.GetID(),
      'ThresholdValue' : imageThreshold,
      'ThresholdType' : 'Above' if invert else 'Below'
      }
    cliNode = slicer.cli.run(slicer.modules.thresholdscalarvolume, None, cliParams, wait_for_completion=True, update_display=showResult)
    # We don't need the CLI module node anymore, remove it to not clutter the scene with it
    slicer.mrmlScene.RemoveNode(cliNode)

    stopTime = time.time()
    logging.info(f'Processing completed in {stopTime-startTime:.2f} seconds')


#
# TransformManagerTest
#

class TransformManagerTest(ScriptedLoadableModuleTest):
  """
  This is the test case for your scripted module.
  Uses ScriptedLoadableModuleTest base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

  def setUp(self):
    """ Do whatever is needed to reset the state - typically a scene clear will be enough.
    """
    slicer.mrmlScene.Clear()

  def runTest(self):
    """Run as few or as many tests as needed here.
    """
    self.setUp()
    self.test_TransformManager1()

  def test_TransformManager1(self):
    """ Ideally you should have several levels of tests.  At the lowest level
    tests should exercise the functionality of the logic with different inputs
    (both valid and invalid).  At higher levels your tests should emulate the
    way the user would interact with your code and confirm that it still works
    the way you intended.
    One of the most important features of the tests is that it should alert other
    developers when their changes will have an impact on the behavior of your
    module.  For example, if a developer removes a feature that you depend on,
    your test should break so they know that the feature is needed.
    """

    self.delayDisplay("Starting the test")

    # Get/create input data

    import SampleData
    registerSampleData()
    inputVolume = SampleData.downloadSample('TransformManager1')
    self.delayDisplay('Loaded test data set')

    inputScalarRange = inputVolume.GetImageData().GetScalarRange()
    self.assertEqual(inputScalarRange[0], 0)
    self.assertEqual(inputScalarRange[1], 695)

    outputVolume = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode")
    threshold = 100

    # Test the module logic

    logic = TransformManagerLogic()

    # Test algorithm with non-inverted threshold
    logic.process(inputVolume, outputVolume, threshold, True)
    outputScalarRange = outputVolume.GetImageData().GetScalarRange()
    self.assertEqual(outputScalarRange[0], inputScalarRange[0])
    self.assertEqual(outputScalarRange[1], threshold)

    # Test algorithm with inverted threshold
    logic.process(inputVolume, outputVolume, threshold, False)
    outputScalarRange = outputVolume.GetImageData().GetScalarRange()
    self.assertEqual(outputScalarRange[0], inputScalarRange[0])
    self.assertEqual(outputScalarRange[1], inputScalarRange[1])

    self.delayDisplay('Test passed')
