"""Tests for the Mesh class."""

import numpy as np
import pytest
from phasta.core.mesh import Mesh


def test_mesh_initialization():
    """Test basic mesh initialization."""
    # Create a simple 2D mesh
    points = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0]
    ])
    
    cells = {
        "quad": np.array([[0, 1, 2, 3]])
    }
    
    # Create mesh
    mesh = Mesh(points, cells)
    
    # Check basic properties
    assert mesh.points.shape == (4, 3)
    assert "quad" in mesh.cells
    assert mesh.cells["quad"].shape == (1, 4)
    
    # Check boundary conditions
    assert isinstance(mesh.boundary_conditions, dict)
    assert len(mesh.boundary_conditions) == 0


def test_mesh_validation():
    """Test mesh validation."""
    # Test invalid points
    with pytest.raises(TypeError):
        Mesh("invalid", {"quad": np.array([[0, 1, 2, 3]])})
    
    with pytest.raises(ValueError):
        Mesh(np.array([1, 2, 3]), {"quad": np.array([[0, 1, 2, 3]])})
    
    # Test invalid cells
    with pytest.raises(TypeError):
        Mesh(np.zeros((4, 3)), "invalid")
    
    with pytest.raises(TypeError):
        Mesh(np.zeros((4, 3)), {"quad": "invalid"})
    
    with pytest.raises(ValueError):
        Mesh(np.zeros((4, 3)), {"quad": np.array([0, 1, 2, 3])})


def test_mesh_boundary_conditions():
    """Test mesh boundary conditions."""
    points = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0]
    ])
    
    cells = {
        "quad": np.array([[0, 1, 2, 3]])
    }
    
    boundary_conditions = {
        "wall": [0, 1],
        "inlet": [1, 2],
        "outlet": [2, 3],
        "symmetry": [3, 0]
    }
    
    mesh = Mesh(points, cells, boundary_conditions)
    
    assert len(mesh.boundary_conditions) == 4
    assert "wall" in mesh.boundary_conditions
    assert "inlet" in mesh.boundary_conditions
    assert "outlet" in mesh.boundary_conditions
    assert "symmetry" in mesh.boundary_conditions
    assert len(mesh.boundary_conditions["wall"]) == 2


def test_aspect_ratio():
    """Test aspect ratio quality metric."""
    # Create quality metric
    metric = AspectRatio()
    
    # Create test mesh
    vertices = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]
    ])
    
    elements = np.array([
        [0, 1, 2, 3]  # Tetrahedron
    ])
    
    # Compute quality
    quality = metric.compute_quality(vertices, elements)
    
    # Check quality
    assert quality > 0  # Quality should be positive
    assert quality < 2  # For regular tetrahedron, aspect ratio should be close to 1
    
    # Test with degenerate element
    vertices_degenerate = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],  # Degenerate vertex
        [0.0, 0.0, 1.0]
    ])
    
    # Should handle zero edge length gracefully
    quality_degenerate = metric.compute_quality(vertices_degenerate, elements)
    assert np.isfinite(quality_degenerate)


def test_multi_resolution_mesh():
    """Test multi-resolution mesh generator."""
    # Create mesh generator
    generator = MultiResolutionMesh(
        base_resolution=1.0,
        refinement_levels=2,
        quality_threshold=2.0
    )
    
    # Create test domain
    domain = np.array([0.0, 0.0, 0.0, 2.0, 2.0, 2.0])
    
    # Generate mesh
    vertices, elements = generator.generate_mesh(domain)
    
    # Check mesh
    assert len(vertices) > 0
    assert len(elements) > 0
    assert vertices.shape[1] == 3  # 3D coordinates
    assert elements.shape[1] == 3  # Triangles
    
    # Test with feature points
    feature_points = np.array([
        [0.5, 0.5, 0.5],
        [1.5, 1.5, 1.5]
    ])
    
    vertices_feature, elements_feature = generator.generate_mesh(domain, feature_points)
    
    # Check that feature points caused refinement
    assert len(vertices_feature) > len(vertices)
    assert len(elements_feature) > len(elements)
    
    # Test edge cases
    with pytest.raises(ValueError):
        generator.generate_mesh(np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0]))  # Invalid domain
    
    # Test with empty feature points
    vertices_empty, elements_empty = generator.generate_mesh(domain, np.array([]))
    assert len(vertices_empty) == len(vertices)
    assert len(elements_empty) == len(elements)


def test_point_cloud_mesh():
    """Test point cloud mesh generator."""
    # Create mesh generator
    generator = PointCloudMesh(
        max_distance=1.0,
        min_angle=20.0
    )
    
    # Create test point cloud
    points = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 1.0, 0.0],
        [1.0, 0.0, 1.0],
        [0.0, 1.0, 1.0],
        [1.0, 1.0, 1.0]
    ])
    
    # Generate mesh
    vertices, elements = generator.generate_mesh(points)
    
    # Check mesh
    assert len(vertices) == len(points)
    assert len(elements) > 0
    assert vertices.shape[1] == 3  # 3D coordinates
    assert elements.shape[1] == 3  # Triangles
    
    # Test with normals
    normals = np.array([
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0]
    ])
    
    vertices_normals, elements_normals = generator.generate_mesh(points, normals)
    
    # Check that normals affected mesh generation
    assert len(elements_normals) != len(elements)
    
    # Test edge cases
    with pytest.raises(ValueError):
        generator.generate_mesh(np.array([[0.0, 0.0, 0.0]]))  # Too few points
    
    # Test with collinear points
    points_collinear = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [2.0, 0.0, 0.0]
    ])
    
    vertices_collinear, elements_collinear = generator.generate_mesh(points_collinear)
    assert len(elements_collinear) == 0  # No valid triangles


def test_memory_management():
    """Test memory management during mesh generation."""
    # Create large point cloud
    n_points = 1000
    points = np.random.rand(n_points, 3)
    
    # Create mesh generators
    multi_res = MultiResolutionMesh()
    point_cloud = PointCloudMesh()
    
    # Test multi-resolution mesh
    domain = np.array([0.0, 0.0, 0.0, 10.0, 10.0, 10.0])
    vertices_multi, elements_multi = multi_res.generate_mesh(domain)
    
    assert len(vertices_multi) > 0
    assert len(elements_multi) > 0
    
    # Test point cloud mesh
    vertices_cloud, elements_cloud = point_cloud.generate_mesh(points)
    
    assert len(vertices_cloud) == n_points
    assert len(elements_cloud) > 0


def test_convergence():
    """Test mesh convergence with refinement."""
    # Create mesh generator
    generator = MultiResolutionMesh(
        base_resolution=1.0,
        refinement_levels=3,
        quality_threshold=2.0
    )
    
    # Create test domain
    domain = np.array([0.0, 0.0, 0.0, 2.0, 2.0, 2.0])
    
    # Generate meshes with different refinement levels
    vertices_1, elements_1 = generator.generate_mesh(domain)
    generator.refinement_levels = 4
    vertices_2, elements_2 = generator.generate_mesh(domain)
    
    # Check that refinement increased mesh size
    assert len(vertices_2) > len(vertices_1)
    assert len(elements_2) > len(elements_1)
    
    # Check that quality improved
    metric = AspectRatio()
    quality_1 = metric.compute_quality(vertices_1, elements_1)
    quality_2 = metric.compute_quality(vertices_2, elements_2)
    
    assert quality_2 <= quality_1  # Quality should improve or stay the same 