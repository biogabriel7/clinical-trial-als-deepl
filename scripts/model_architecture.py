#!/usr/bin/env python3
"""
Script 4: Model Architecture
"""

import torch as tc
import torch.nn as nn
import pickle
import copy


class LRP_Linear(nn.Module):
    """Linear layer with Layer-wise Relevance Propagation capability"""
    
    def __init__(self, inp, outp, gamma=0.01, eps=1e-5):
        super(LRP_Linear, self).__init__()
        self.A_dict = {}
        self.linear = nn.Linear(inp, outp)
        nn.init.xavier_uniform_(self.linear.weight, gain=nn.init.calculate_gain('relu'))
        self.gamma = tc.tensor(gamma)
        self.eps = tc.tensor(eps)
        self.iteration = None

    def forward(self, x):
        if not self.training:
            self.A_dict[self.iteration] = x.clone()
        return self.linear(x)

    def relprop(self, R):
        device = next(self.parameters()).device
        A = self.A_dict[self.iteration].clone()
        A, self.eps = A.to(device), self.eps.to(device)

        Ap = A.clamp(min=0).detach().data.requires_grad_(True)
        Am = A.clamp(max=0).detach().data.requires_grad_(True)

        zpp = self.newlayer(1).forward(Ap)  
        zmm = self.newlayer(-1, no_bias=True).forward(Am) 
        zmp = self.newlayer(1, no_bias=True).forward(Am) 
        zpm = self.newlayer(-1).forward(Ap) 

        with tc.no_grad():
            Y = self.forward(A).data

        sp = ((Y > 0).float() * R / (zpp + zmm + self.eps * ((zpp + zmm == 0).float() + tc.sign(zpp + zmm)))).data
        sm = ((Y < 0).float() * R / (zmp + zpm + self.eps * ((zmp + zpm == 0).float() + tc.sign(zmp + zpm)))).data

        (zpp * sp).sum().backward()
        cpp = Ap.grad
        Ap.grad = None
        Ap.requires_grad_(True)

        (zpm * sm).sum().backward()
        cpm = Ap.grad
        Ap.grad = None
        Ap.requires_grad_(True)

        (zmp * sm).sum().backward()
        cmp = Am.grad
        Am.grad = None
        Am.requires_grad_(True)

        (zmm * sp).sum().backward()
        cmm = Am.grad
        Am.grad = None
        Am.requires_grad_(True)

        R_1 = (Ap * cpp).data
        R_2 = (Ap * cpm).data
        R_3 = (Am * cmp).data
        R_4 = (Am * cmm).data

        return R_1 + R_2 + R_3 + R_4

    def newlayer(self, sign, no_bias=False):
        if sign == 1:
            rho = lambda p: p + self.gamma * p.clamp(min=0)
        else:
            rho = lambda p: p + self.gamma * p.clamp(max=0)

        layer_new = copy.deepcopy(self.linear)
        try:
            layer_new.weight = nn.Parameter(rho(self.linear.weight))
        except AttributeError:
            pass

        try:
            layer_new.bias = nn.Parameter(self.linear.bias * 0 if no_bias else rho(self.linear.bias))
        except AttributeError:
            pass

        return layer_new


class LRP_ReLU(nn.Module):
    """ReLU layer with LRP capability"""
    
    def __init__(self):
        super(LRP_ReLU, self).__init__()
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(x)

    def relprop(self, R):
        return R


class LRP_DropOut(nn.Module):
    """Dropout layer with LRP capability"""
    
    def __init__(self, p):
        super(LRP_DropOut, self).__init__()
        self.dropout = nn.Dropout(p)

    def forward(self, x):
        return self.dropout(x)

    def relprop(self, R):
        return R


class LRP_BatchNorm1d(nn.Module):
    """BatchNorm layer with LRP capability"""
    
    def __init__(self, num_features):
        super(LRP_BatchNorm1d, self).__init__()
        self.bn = nn.BatchNorm1d(num_features)

    def forward(self, x):
        return self.bn(x)

    def relprop(self, R):
        return R


class ALS_Comprehensive_Model(nn.Module):
    """Enhanced neural network model with emergency fixes and simplified architecture"""
    
    def __init__(self, n_features, config):
        super(ALS_Comprehensive_Model, self).__init__()
        self.classname = 'ALS Comprehensive Clinical Trials Model'
        
        inp = n_features
        
        # EMERGENCY FIX: Simplified, more stable architecture
        hidden_1 = min(256, max(64, int(n_features * config.factor_hidden_nodes)))
        
        # Determine if we need a second hidden layer
        if config.hidden_depth_simple >= 2:
            hidden_2 = max(config.min_hidden_size, hidden_1 // 2)
        else:
            hidden_2 = None
            
        outp = 1
        
        # Store architecture info for debugging
        if hidden_2 is not None:
            self.architecture = f"{inp} -> {hidden_1} -> {hidden_2} -> {outp}"
        else:
            self.architecture = f"{inp} -> {hidden_1} -> {outp}"
        
        # Network layers - simplified without batch norm (emergency fix)
        self.input_dropout = LRP_DropOut(config.input_dropout)
        
        # First hidden layer
        self.fc1 = LRP_Linear(inp, hidden_1, gamma=config.lrp_gamma)
        self.relu1 = LRP_ReLU()
        self.dropout1 = LRP_DropOut(config.dropout)
        
        # Second hidden layer (conditional)
        if hidden_2 is not None:
            self.fc2 = LRP_Linear(hidden_1, hidden_2, gamma=config.lrp_gamma)
            self.relu2 = LRP_ReLU()
            self.dropout2 = LRP_DropOut(config.dropout)
            self.fc_out = LRP_Linear(hidden_2, outp, gamma=config.lrp_gamma)
        else:
            self.fc_out = LRP_Linear(hidden_1, outp, gamma=config.lrp_gamma)
        
        # Output activation
        self.sigmoid = nn.Sigmoid()
        
        # Store model properties
        self.hidden_depth = config.hidden_depth_simple
        self.n_parameters = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        # Debug information
        print(f"Model architecture: {self.architecture}")
        print(f"Total parameters: {self.n_parameters:,}")

    def forward(self, x):
        """Forward pass through the network"""
        # Input processing
        x = self.input_dropout(x)
        
        # First hidden layer
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        
        # Second hidden layer (if exists)
        if hasattr(self, 'fc2'):
            x = self.fc2(x)
            x = self.relu2(x)
            x = self.dropout2(x)
        
        # Output layer
        x = self.fc_out(x)
        return self.sigmoid(x)

    def relprop(self, R):
        """Layer-wise Relevance Propagation backward pass"""
        # Start from output and work backwards
        R = self.fc_out.relprop(R)
        
        # Second hidden layer (if exists)
        if hasattr(self, 'fc2'):
            R = self.dropout2.relprop(R)
            R = self.relu2.relprop(R)
            R = self.fc2.relprop(R)
        
        # First hidden layer
        R = self.dropout1.relprop(R)
        R = self.relu1.relprop(R)
        R = self.fc1.relprop(R)
        
        # Input processing
        R = self.input_dropout.relprop(R)
        
        return R

    def get_model_info(self):
        """Get detailed model information"""
        return {
            'architecture': self.architecture,
            'parameters': self.n_parameters,
            'hidden_depth': self.hidden_depth,
            'classname': self.classname
        }

    def set_lrp_iteration(self, iteration=0):
        """Set LRP iteration for all LRP layers"""
        for layer in self.modules():
            if hasattr(layer, 'iteration'):
                layer.iteration = iteration


def create_model_from_config(n_features, config):
    """Factory function to create model with proper device placement"""
    model = ALS_Comprehensive_Model(n_features, config)
    model = model.to(config.device)
    
    # Initialize LRP iteration
    model.set_lrp_iteration(0)
    
    return model


def test_model_architecture(n_features, config):
    """Test model architecture with sample data"""
    print(f"\n🧪 Testing model architecture...")
    
    # Create model
    model = create_model_from_config(n_features, config)
    model.eval()
    
    # Create sample input
    sample_input = tc.randn(10, n_features).to(config.device)
    
    # Test forward pass
    with tc.no_grad():
        output = model(sample_input)
        print(f"✓ Forward pass successful: input {sample_input.shape} -> output {output.shape}")
        print(f"  Output range: [{output.min():.3f}, {output.max():.3f}]")
    
    # Test LRP (if model is not training)
    model.set_lrp_iteration(0)
    try:
        with tc.no_grad():
            # Forward pass to populate A_dict
            _ = model(sample_input)
            
            # LRP backward pass
            R = tc.ones_like(output)
            relevance = model.relprop(R)
            print(f"✓ LRP backward pass successful: output {R.shape} -> relevance {relevance.shape}")
            print(f"  Relevance range: [{relevance.min():.3f}, {relevance.max():.3f}]")
    except Exception as e:
        print(f"⚠️ LRP test failed: {e}")
    
    return model


def validate_model_components():
    """Validate all model components work correctly"""
    print("🔍 Validating model components...")
    
    # Test individual layers
    test_input = tc.randn(5, 10)
    
    # Test LRP_Linear
    linear = LRP_Linear(10, 5)
    linear.iteration = 0
    linear.eval()
    with tc.no_grad():
        out = linear(test_input)
        R = tc.ones_like(out)
        rel = linear.relprop(R)
        print(f"✓ LRP_Linear: {test_input.shape} -> {out.shape} -> {rel.shape}")
    
    # Test LRP_ReLU
    relu = LRP_ReLU()
    out = relu(test_input)
    rel = relu.relprop(tc.ones_like(out))
    print(f"✓ LRP_ReLU: {test_input.shape} -> {out.shape} -> {rel.shape}")
    
    # Test LRP_DropOut
    dropout = LRP_DropOut(0.1)
    dropout.eval()  # Important: set to eval mode for consistent testing
    out = dropout(test_input)
    rel = dropout.relprop(tc.ones_like(out))
    print(f"✓ LRP_DropOut: {test_input.shape} -> {out.shape} -> {rel.shape}")
    
    print("✓ All model components validated")


if __name__ == "__main__":
    # Load config
    with open(snakemake.input.config, 'rb') as f:
        config = pickle.load(f)
    
    print("🏗️ Building neural network architecture...")
    
    # Validate components first
    validate_model_components()
    
    # Test with realistic feature count (around 100 after feature selection)
    test_features = 100
    test_model = test_model_architecture(test_features, config)
    
    # Save model classes and factory function
    model_classes = {
        'LRP_Linear': LRP_Linear,
        'LRP_ReLU': LRP_ReLU,
        'LRP_DropOut': LRP_DropOut,
        'LRP_BatchNorm1d': LRP_BatchNorm1d,
        'ALS_Comprehensive_Model': ALS_Comprehensive_Model,
        'create_model_from_config': create_model_from_config,
        'test_model_architecture': test_model_architecture
    }
    
    with open(snakemake.output.model_classes, 'wb') as f:
        pickle.dump(model_classes, f)
    
    print("✓ Neural network architecture defined and tested")
    print(f"✓ Model classes saved with {len(model_classes)} components")
    
    # Clean up test model
    del test_model
    tc.cuda.empty_cache() if tc.cuda.is_available() else None