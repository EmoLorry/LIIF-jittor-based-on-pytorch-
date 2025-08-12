#!/usr/bin/env python3
"""
Test script for the unified models.py
Tests if all models can be created successfully
"""

import torch
import models
import utils

def test_edsr_baseline():
    """Test EDSR-baseline creation"""
    print("Testing EDSR-baseline...")
    
    config = {
        'name': 'edsr-baseline',
        'args': {
            'n_resblocks': 16,
            'n_feats': 64, 
            'res_scale': 1,
            'scale': 2,
            'no_upsampling': True,  # Important for LIIF
            'rgb_range': 1
        }
    }
    
    try:
        encoder = models.make(config)
        print(f"  ✓ EDSR-baseline created successfully")
        print(f"  ✓ Output dimension: {encoder.out_dim}")
        
        # Test forward pass
        x = torch.randn(1, 3, 48, 48)
        feat = encoder(x)
        print(f"  ✓ Forward pass successful: {x.shape} -> {feat.shape}")
        return encoder
    except Exception as e:
        print(f"  ✗ EDSR-baseline failed: {e}")
        return None


def test_mlp():
    """Test MLP creation"""
    print("\nTesting MLP...")
    
    config = {
        'name': 'mlp',
        'args': {
            'in_dim': 580,  # 64*9 + 2 + 2 for LIIF
            'out_dim': 3,
            'hidden_list': [256, 256, 256, 256]
        }
    }
    
    try:
        mlp = models.make(config)
        print(f"  ✓ MLP created successfully")
        
        # Test forward pass
        x = torch.randn(16, 2304, 580)  # batch_size, num_queries, input_dim
        output = mlp(x)
        print(f"  ✓ Forward pass successful: {x.shape} -> {output.shape}")
        return mlp
    except Exception as e:
        print(f"  ✗ MLP failed: {e}")
        return None


def test_liif():
    """Test LIIF main model creation"""
    print("\nTesting LIIF...")
    
    config = {
        'name': 'liif',
        'args': {
            'encoder_spec': {
                'name': 'edsr-baseline',
                'args': {
                    'n_resblocks': 16,
                    'n_feats': 64,
                    'res_scale': 1,
                    'scale': 2,
                    'no_upsampling': True,
                    'rgb_range': 1
                }
            },
            'imnet_spec': {
                'name': 'mlp',
                'args': {
                    'out_dim': 3,
                    'hidden_list': [256, 256, 256, 256]
                }
            },
            'local_ensemble': True,
            'feat_unfold': True,
            'cell_decode': True
        }
    }
    
    try:
        liif_model = models.make(config)
        print(f"  ✓ LIIF created successfully")
        
        # Test forward pass with small inputs
        inp = torch.randn(1, 3, 48, 48)
        coord = torch.randn(1, 100, 2)  # 100 query points
        cell = torch.randn(1, 100, 2)
        
        output = liif_model(inp, coord, cell)
        print(f"  ✓ Forward pass successful: {inp.shape} -> {output.shape}")
        print(f"  ✓ Query {coord.shape[1]} points -> {output.shape[1]} RGB values")
        return liif_model
    except Exception as e:
        print(f"  ✗ LIIF failed: {e}")
        return None


def test_rdn():
    """Test RDN creation"""
    print("\nTesting RDN...")
    
    config = {
        'name': 'rdn',
        'args': {
            'G0': 64,
            'RDNkSize': 3,
            'RDNconfig': 'B',
            'scale': 2,
            'no_upsampling': True  # Important for LIIF
        }
    }
    
    try:
        encoder = models.make(config)
        print(f"  ✓ RDN created successfully")
        print(f"  ✓ Output dimension: {encoder.out_dim}")
        
        # Test forward pass
        x = torch.randn(1, 3, 48, 48)
        feat = encoder(x)
        print(f"  ✓ Forward pass successful: {x.shape} -> {feat.shape}")
        return encoder
    except Exception as e:
        print(f"  ✗ RDN failed: {e}")
        return None


def test_metasr():
    """Test MetaSR creation"""
    print("\nTesting MetaSR...")
    
    config = {
        'name': 'metasr',
        'args': {
            'encoder_spec': {
                'name': 'edsr-baseline',
                'args': {
                    'n_resblocks': 16,
                    'n_feats': 64,
                    'res_scale': 1,
                    'scale': 2,
                    'no_upsampling': True,
                    'rgb_range': 1
                }
            }
        }
    }
    
    try:
        metasr_model = models.make(config)
        print(f"  ✓ MetaSR created successfully")
        
        # Test forward pass with small inputs
        inp = torch.randn(1, 3, 48, 48)
        coord = torch.randn(1, 100, 2)  # 100 query points
        cell = torch.randn(1, 100, 2)
        
        output = metasr_model(inp, coord, cell)
        print(f"  ✓ Forward pass successful: {inp.shape} -> {output.shape}")
        print(f"  ✓ Query {coord.shape[1]} points -> {output.shape[1]} RGB values")
        return metasr_model
    except Exception as e:
        print(f"  ✗ MetaSR failed: {e}")
        return None


def test_model_registration():
    """Test model registration system"""
    print("\nTesting model registration...")
    
    registered_models = list(models.models.keys())
    print(f"  ✓ Registered models: {registered_models}")
    
    expected_models = ['mlp', 'liif', 'edsr-baseline', 'edsr', 'rdn', 'metasr']
    missing_models = [m for m in expected_models if m not in registered_models]
    
    if missing_models:
        print(f"  ✗ Missing models: {missing_models}")
    else:
        print(f"  ✓ All expected models registered")


if __name__ == "__main__":
    print("=" * 50)
    print("Testing Unified Models.py")
    print("=" * 50)
    
    test_model_registration()
    encoder = test_edsr_baseline()
    mlp = test_mlp()
    liif_model = test_liif()
    rdn_model = test_rdn()
    metasr_model = test_metasr()
    
    print("\n" + "=" * 50)
    if encoder and mlp and liif_model and rdn_model and metasr_model:
        print("🎉 ALL TESTS PASSED! Your unified models.py is working correctly!")
        print("✅ Successfully integrated: LIIF, EDSR, MLP, RDN, MetaSR")
    else:
        print("❌ Some tests failed. Check the error messages above.")
    print("=" * 50)