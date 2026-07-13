"""Zoom/navigation service for recursive nebula system."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Any

from server.models.cloud import Cloud, CloudPlacement
from server.models.space import Space
from server.repositories.cloud_repository import (
    CloudRepository, CloudPlacementRepository, SpaceRepository, StructuralComponentRepository
)


@dataclass
class ZoomPath:
    """Breadcrumb path for zoom navigation."""
    spaces: List[Dict[str, Any]] = None
    current_space_id: Optional[int] = None
    mode: str = "structural"  # structural or semantic
    
    def __post_init__(self):
        if self.spaces is None:
            self.spaces = []


class ZoomService:
    """Handles zoom in/out navigation between nebula spaces."""
    
    def __init__(self):
        self.cloud_repo = CloudRepository()
        self.placement_repo = CloudPlacementRepository()
        self.space_repo = SpaceRepository()
        self.component_repo = StructuralComponentRepository()
        
        # Active zoom paths per session
        self.paths: Dict[str, ZoomPath] = {}
    
    def get_or_create_path(self, session_id: str) -> ZoomPath:
        if session_id not in self.paths:
            self.paths[session_id] = ZoomPath()
        return self.paths[session_id]
    
    def zoom_in_structural(self, session_id: str, cloud_id: int) -> Optional[Dict[str, Any]]:
        """
        Zoom into a cloud's structural space (lower layer composition).
        Returns space info with child clouds.
        """
        path = self.get_or_create_path(session_id)
        
        # Get cloud
        cloud = self.cloud_repo.get_by_id(cloud_id)
        if not cloud:
            return None
        
        # Get or create structural space
        structural_space = self.space_repo.get_structural_space(cloud_id)
        if not structural_space:
            # Create structural space for this cloud
            structural_space = Space(
                host_cloud_id=cloud_id,
                layer_id=cloud.layer_id - 1,  # lower layer
                mode="structural",
                coordinate_dimensions=2,
                scale=1.0,
            )
            structural_space = self.space_repo.create(structural_space)
            
            # Populate with structural components
            self._populate_structural_space(structural_space, cloud)
        
        # Get placements in this space
        placements = self.placement_repo.get_by_space(structural_space.id)
        
        # Load cloud data for each placement
        children = []
        for p in placements:
            child_cloud = self.cloud_repo.get_by_id(p.cloud_id)
            if child_cloud:
                children.append({
                    "cloud": child_cloud.to_dict(),
                    "placement": p.to_dict(),
                })
        
        # Add to breadcrumb path
        path.spaces.append({
            "space_id": structural_space.id,
            "host_cloud_id": cloud_id,
            "mode": "structural",
            "layer": cloud.layer_id - 1,
        })
        path.current_space_id = structural_space.id
        path.mode = "structural"
        
        return {
            "space": structural_space.to_dict(),
            "host_cloud": cloud.to_dict(),
            "children": children,
            "breadcrumb": path.spaces,
        }
    
    def zoom_in_semantic(self, session_id: str, cloud_id: int) -> Optional[Dict[str, Any]]:
        """
        Zoom into a cloud's semantic space (same layer neighbors).
        Returns space info with neighboring/projected clouds.
        """
        path = self.get_or_create_path(session_id)
        
        cloud = self.cloud_repo.get_by_id(cloud_id)
        if not cloud:
            return None
        
        # Get or create semantic space
        semantic_space = self.space_repo.get_semantic_space(cloud_id)
        if not semantic_space:
            # Create semantic space
            semantic_space = Space(
                host_cloud_id=cloud_id,
                layer_id=cloud.layer_id,
                mode="semantic",
                coordinate_dimensions=2,
                scale=1.0,
            )
            semantic_space = self.space_repo.create(semantic_space)
            
            # Populate with semantic projections
            self._populate_semantic_space(semantic_space, cloud)
        
        # A semantic space may have been created by training before it was
        # first opened. Populate missing projections lazily on first access.
        self._populate_semantic_space(semantic_space, cloud)

        # Get placements
        placements = self.placement_repo.get_by_space(semantic_space.id)
        
        neighbors = []
        for p in placements:
            if p.cloud_id == cloud_id:
                continue  # skip host
            neighbor_cloud = self.cloud_repo.get_by_id(p.cloud_id)
            if neighbor_cloud:
                neighbors.append({
                    "cloud": neighbor_cloud.to_dict(),
                    "placement": p.to_dict(),
                })
        
        # Add to breadcrumb
        path.spaces.append({
            "space_id": semantic_space.id,
            "host_cloud_id": cloud_id,
            "mode": "semantic",
            "layer": cloud.layer_id,
        })
        path.current_space_id = semantic_space.id
        path.mode = "semantic"
        
        return {
            "space": semantic_space.to_dict(),
            "host_cloud": cloud.to_dict(),
            "neighbors": neighbors,
            "breadcrumb": path.spaces,
        }
    
    def zoom_out(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Zoom out to parent space."""
        path = self.get_or_create_path(session_id)
        
        if len(path.spaces) <= 1:
            # At root, return to global layer space
            path.spaces = []
            path.current_space_id = None
            return {"at_root": True, "breadcrumb": []}
        
        # Pop current space
        path.spaces.pop()
        
        if path.spaces:
            parent_space_info = path.spaces[-1]
            path.current_space_id = parent_space_info["space_id"]
            path.mode = parent_space_info["mode"]
            
            # Load parent space data
            parent_space = self.space_repo.get_by_id(path.current_space_id)
            if parent_space:
                host_cloud = self.cloud_repo.get_by_id(parent_space.host_cloud_id)
                placements = self.placement_repo.get_by_space(path.current_space_id)
                
                items = []
                for p in placements:
                    c = self.cloud_repo.get_by_id(p.cloud_id)
                    if c:
                        items.append({
                            "cloud": c.to_dict(),
                            "placement": p.to_dict(),
                        })
                
                return {
                    "space": parent_space.to_dict(),
                    "host_cloud": host_cloud.to_dict() if host_cloud else None,
                    "items": items,
                    "breadcrumb": path.spaces,
                }
        
        return {"at_root": True, "breadcrumb": []}
    
    def get_current_space(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get current space view."""
        path = self.get_or_create_path(session_id)
        
        if not path.current_space_id:
            return {"at_root": True, "breadcrumb": path.spaces}
        
        space = self.space_repo.get_by_id(path.current_space_id)
        if not space:
            return {"at_root": True, "breadcrumb": path.spaces}
        
        placements = self.placement_repo.get_by_space(path.current_space_id)
        items = []
        for p in placements:
            c = self.cloud_repo.get_by_id(p.cloud_id)
            if c:
                items.append({
                    "cloud": c.to_dict(),
                    "placement": p.to_dict(),
                })
        
        return {
            "space": space.to_dict(),
            "host_cloud": self.cloud_repo.get_by_id(space.host_cloud_id).to_dict() if space.host_cloud_id else None,
            "items": items,
            "breadcrumb": path.spaces,
        }
    
    def _populate_structural_space(self, space: Space, parent_cloud: Cloud) -> None:
        """Fill structural space with child components."""
        components = self.component_repo.get_children(parent_cloud.id)
        
        for idx, comp in enumerate(components):
            child_cloud = self.cloud_repo.get_by_id(comp.child_cloud_id)
            if not child_cloud:
                continue
            
            # Create or get placement
            placement = self.placement_repo.get_by_id(comp.child_placement_id) if comp.child_placement_id else None
            
            if not placement:
                # Position in a line
                x = 200 + idx * 150
                y = 300
                placement = self.placement_repo.create(CloudPlacement(
                    space_id=space.id,
                    cloud_id=child_cloud.id,
                    x=x,
                    y=y,
                    radius=child_cloud.radius,
                    density=child_cloud.density,
                    mass=child_cloud.mass,
                    activation=0.0,
                ))
                # Update component with placement ID
                # (would need update method)
            
            # Update component
            comp.child_placement_id = placement.id
            comp.position_index = idx
    
    def _populate_semantic_space(self, space: Space, host_cloud: Cloud) -> None:
        """Fill semantic space with related concept projections."""
        # Get co-activation neighbors
        from server.services.activation import get_coactivation_neighbors
        
        neighbors = get_coactivation_neighbors(host_cloud.id, host_cloud.layer_id, min_score=0.05, limit=20)
        
        # Also get clouds in same layer
        layer_clouds = self.cloud_repo.get_by_layer(host_cloud.layer_id, limit=100)
        
        # Combine and deduplicate
        all_clouds = {host_cloud.id: host_cloud}
        for nid, score in neighbors:
            cloud = self.cloud_repo.get_by_id(nid)
            if cloud:
                all_clouds[nid] = cloud
        
        # Add some random same-layer clouds for context
        for c in layer_clouds[:30]:
            if c.id not in all_clouds:
                all_clouds[c.id] = c
        
        existing_ids = {
            placement.cloud_id
            for placement in self.placement_repo.get_by_space(space.id)
        }

        # Place them in semantic space around host
        import math
        for idx, (cid, cloud) in enumerate(all_clouds.items()):
            if cid in existing_ids:
                continue
            if cid == host_cloud.id:
                # Host at center
                x, y = 400, 300
            else:
                # Arrange in circle
                angle = (idx * 2 * math.pi) / max(1, len(all_clouds) - 1)
                radius = 250 + (idx % 3) * 50
                x = 400 + radius * math.cos(angle)
                y = 300 + radius * math.sin(angle)
            
            placement = self.placement_repo.create(CloudPlacement(
                space_id=space.id,
                cloud_id=cloud.id,
                x=x,
                y=y,
                radius=cloud.radius,
                density=cloud.density,
                mass=cloud.mass,
                activation=0.0,
            ))


# Global instance
zoom_service = ZoomService()
