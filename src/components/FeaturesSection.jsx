import React from 'react';
import { ScanFace, Activity, Zap, Eye, ShieldCheck, CheckCircle2 } from 'lucide-react';

export default function FeaturesSection() {
  const features = [
    {
      icon: <ScanFace size={24} />,
      title: 'Facial Extraction & Normalization',
      description:
        'Isolates facial regions across uniform video keyframes using multi-orientation OpenCV cascades, applying square geometry preservation for neural preprocessing.'
    },
    {
      icon: <Activity size={24} />,
      title: 'Vision Transformer (ViT) Architecture',
      description:
        'Leverages a fine-tuned Vision Transformer (google/vit-base-patch16-224) trained on 140,000 real and manipulated facial images to detect generative synthesis patterns.'
    },
    {
      icon: <Zap size={24} />,
      title: 'Multi-Frame Score Aggregation',
      description:
        'Computes frame-by-frame softmax logits, evaluating mean fake probability, peak manipulation scores, and inter-frame variance to prevent single-frame anomalies.'
    },
    {
      icon: <ShieldCheck size={24} />,
      title: 'FastAPI Production Backend',
      description:
        'Executes PyTorch neural inference directly on server hardware, reporting genuine execution latencies and unaltered JSON payloads without simulation.'
    }
  ];

  return (
    <section className="features-section">
      <div className="container">
        <div className="features-header">
          <h2>Neural Detection Architecture</h2>
          <p>
            TruthLens AI combines OpenCV facial extraction with pretrained Vision Transformer 
            classification for empirical deepfake video verification.
          </p>
        </div>

        <div className="features-grid">
          {features.map((feat, index) => (
            <div key={index} className="feature-card">
              <div className="feature-icon-box">{feat.icon}</div>
              <h3 className="feature-title">{feat.title}</h3>
              <p className="feature-desc">{feat.description}</p>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}
