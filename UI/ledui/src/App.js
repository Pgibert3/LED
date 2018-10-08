import React, { Component } from 'react';
import AnimationCreator from './components/AnimationCreator'

class App extends Component {
    constructor(props) {
        super(props);
    }

    render() {
        return(
            <AnimationCreator title = "Animation Creator" />
        );
    }
}

export default App;
