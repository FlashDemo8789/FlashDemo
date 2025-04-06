"""
Streamlit-PyTorch Compatibility Module

This module resolves known conflicts between PyTorch and Streamlit's file watcher
by implementing a patch that prevents Streamlit from watching PyTorch-related modules.

Usage:
    import torch_streamlit_fix
    torch_streamlit_fix.apply_patch()
"""

import os
import sys
import logging
import importlib
import asyncio
import traceback
from typing import List, Any, Optional, Callable, Dict

# Configure module logger
logger = logging.getLogger(__name__)

class StreamlitPatcher:
    """
    Patches Streamlit's file watcher to prevent conflicts with PyTorch.
    Implements a robust patching mechanism with fallbacks and verification.
    """
    
    def __init__(self):
        self.applied = False
        self.original_extract_paths = None
        self.patched_modules = []
    
    def _set_environment_variables(self) -> None:
        """Set environment variables to exclude PyTorch modules from file watching."""
        exclusion_list = [
            "torch", "_torch", "torch._C", "torch.nn", "torch.cuda", 
            "torch.utils", "torch.functional", "torch.classes"
        ]
        os.environ["STREAMLIT_WATCH_FILES_EXCLUDE"] = ",".join(exclusion_list)
        logger.info(f"Set Streamlit exclusion environment variables for torch modules")
    
    def _patch_file_watcher(self) -> bool:
        """
        Patch Streamlit's local_sources_watcher module to avoid PyTorch conflicts.
        
        Returns:
            bool: True if patching was successful, False otherwise
        """
        try:
            # Import the module without triggering the error
            lsw_module = importlib.import_module("streamlit.watcher.local_sources_watcher")
            
            # Save the original function for reference
            self.original_extract_paths = lsw_module.extract_paths
            
            # Define a safer version that handles PyTorch modules
            def safe_extract_paths(module: Any) -> List[str]:
                """
                A safer version of extract_paths that avoids PyTorch modules.
                
                Args:
                    module: The module to extract paths from
                
                Returns:
                    List[str]: List of file paths or empty list for PyTorch modules
                """
                module_name = getattr(module, "__name__", "")
                
                # Skip problematic modules
                if "torch" in module_name:
                    return []
                
                # Call the original function wrapped in a try-except
                try:
                    return self.original_extract_paths(module)
                except Exception as e:
                    logger.warning(f"Error in extract_paths for module {module_name}: {str(e)}")
                    return []
            
            # Replace the original function with our safer version
            lsw_module.extract_paths = safe_extract_paths
            
            self.patched_modules.append("streamlit.watcher.local_sources_watcher")
            logger.info("Successfully patched Streamlit file watcher")
            return True
            
        except ImportError:
            logger.warning("Could not import streamlit.watcher.local_sources_watcher")
            return False
        except Exception as e:
            logger.error(f"Failed to patch file watcher: {str(e)}")
            return False
    
    def _fix_asyncio(self) -> bool:
        """
        Fix asyncio event loop issues on Windows.
        
        Returns:
            bool: True if fix was applied, False otherwise
        """
        try:
            if sys.platform == 'win32':
                asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
                logger.info("Set Windows-compatible asyncio event loop policy")
            return True
        except Exception as e:
            logger.error(f"Failed to set asyncio policy: {str(e)}")
            return False
    
    def apply_patch(self) -> bool:
        """
        Apply all patches and fixes to make Streamlit work with PyTorch.
        
        Returns:
            bool: True if all patches were successfully applied
        """
        if self.applied:
            logger.info("Patch already applied")
            return True
        
        logger.info("Applying PyTorch-Streamlit compatibility patches")
        
        # Step 1: Set environment variables
        self._set_environment_variables()
        
        # Step 2: Apply file watcher patch
        file_watcher_patched = self._patch_file_watcher()
        
        # Step 3: Fix asyncio on Windows
        asyncio_fixed = self._fix_asyncio()
        
        # Step 4: Verify the patch
        patch_verified = self._verify_patch()
        
        # Mark as applied if all steps succeeded
        self.applied = file_watcher_patched and asyncio_fixed and patch_verified
        
        if self.applied:
            logger.info("Successfully applied all PyTorch-Streamlit compatibility patches")
        else:
            logger.warning("Some patches failed to apply - see log for details")
        
        return self.applied
    
    def _verify_patch(self) -> bool:
        """
        Verify that the patch was correctly applied by testing common problematic paths.
        
        Returns:
            bool: True if verification passes
        """
        try:
            # Test the patched function with a dummy torch-like module
            class DummyTorchModule:
                __name__ = "torch.dummy"
                
                def __getattr__(self, name):
                    if name == "__path__":
                        raise RuntimeError("Test exception from __path__")
            
            # Create a dummy module
            dummy_module = DummyTorchModule()
            
            # Get the patched extract_paths function
            try:
                lsw_module = importlib.import_module("streamlit.watcher.local_sources_watcher")
                patched_extract_paths = lsw_module.extract_paths
                
                # Test it - should return empty list without error
                result = patched_extract_paths(dummy_module)
                
                # Verify result is empty list
                if result == []:
                    logger.info("Patch verification passed")
                    return True
                else:
                    logger.warning(f"Patch verification failed: unexpected result {result}")
                    return False
                    
            except Exception as e:
                logger.warning(f"Patch verification failed: {str(e)}")
                return False
            
        except Exception as e:
            logger.error(f"Error during patch verification: {str(e)}")
            return False

# Create a global instance for easy access
patcher = StreamlitPatcher()

def apply_patch() -> bool:
    """
    Apply the PyTorch-Streamlit compatibility patch.
    
    Returns:
        bool: True if patch was applied successfully
    """
    return patcher.apply_patch()

# Apply the patch when the module is imported
apply_patch() 