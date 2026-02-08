# Linear Algebra App

An interactive web application built with Next.js to visualize 2D linear transformations, matrix multiplication, eigenvalues, and singular value decomposition (SVD).

## Features

- **Interactive Visualization**: See how linear transformations affect a 2D grid, unit circle, and basis vectors in real-time.
- **Composite Transformations**: Explore the effect of applying Matrix A followed by Matrix B ($C = B \cdot A$).
- **Matrix Analysis**:
  - **Determinant**: Visualized as the area of the transformed parallelogram.
  - **Eigenvalues & Eigenvectors**: Visual representation of invariant directions.
  - **Eigendecomposition**: Breakdown of the matrix into $PDP^{-1}$.
  - **Singular Value Decomposition (SVD)**: Breakdown into Rotation ($V^T$), Scaling ($\Sigma$), and Rotation ($U$).
- **Animations**:
  - **Transformation Flow**: Animate from Identity $\to$ A $\to$ BA.
  - **SVD Animation**: Visualize the decomposition steps (Rotate $\to$ Scale $\to$ Rotate).
- **Inverse Matrix**: Automatically calculate and apply the inverse transformation ($B = A^{-1}$).
- **Presets**: Quickly apply common transformations like Rotation, Scaling, and Shear.

## Tech Stack

- **Framework**: [Next.js](https://nextjs.org/) (App Router)
- **Language**: [TypeScript](https://www.typescriptlang.org/)
- **Styling**: [Tailwind CSS](https://tailwindcss.com/)
- **Animation**: [Framer Motion](https://www.framer.com/motion/)
- **Icons**: [Lucide React](https://lucide.dev/)

## Getting Started

1.  **Clone the repository**

    ```bash
    git clone <repository-url>
    cd linear-algebra-app
    ```

2.  **Install dependencies**

    ```bash
    npm install
    # or
    yarn install
    # or
    pnpm install
    ```

3.  **Run the development server**

    ```bash
    npm run dev
    # or
    yarn dev
    # or
    pnpm dev
    ```

4.  **Open your browser**

    Navigate to [http://localhost:3000](http://localhost:3000) to see the application.

## Usage

1.  **Input Matrices**: Enter values for Matrix A and Matrix B in the control panel.
2.  **Observe Results**: The grid on the right updates instantly. The "Composite Matrix C" shows the result of $B \cdot A$.
3.  **Play Animations**: Use the "Play Transformation" or "Play SVD" buttons to watch the transformation unfold.
4.  **Analyze**: Check the panel below the visualization for mathematical details like the determinant and eigenvalues.