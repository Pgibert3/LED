import React, { Component } from 'react';

class App extends Component {
    constructor(props) {
        super(props);
    }

    render() {
        const globalFields = ["Name", "ID"];
        const animTypes = ["Push", "Switch"];
        const effectData = {
            "Push":["0", "1"],
            "Switch":["0"]
        };
        return(<AnimationCreator globalFields={globalFields} animTypes={animTypes} effectData={effectData}/>);
    }
}


class AnimationCreator extends Component {
    constructor(props) {
        super(props);
        this.state = {
            animType: "Push",
        }
        this.onAnimTypeChange = this.onAnimTypeChange.bind(this);
        this.onEffectChange = this.onAnimTypeChange.bind(this);
    }

    render() {
        return(
          <div>
            <h1>Create Animation</h1>
                <GlobalFieldList fields={this.props.globalFields} />
                <AnimTypeSelect
                    label="Type"
                    options={this.props.animTypes}
                    onChange={this.onAnimTypeChange}
                    effectData={this.props.effectData}
                />
          </div>
        );
    }

    onAnimTypeChange(e) {
        this.setState({
            animType: e.target.value,
        });
    }

    onEffectChange(e) {

    }
}


function AnimTypeSelect(props) {
    const optionList = props.options.map((option) =>
        <option key={option.toString()} value={option.toString()}>{option.toString()}</option>);
    return(
      <div>
          <label>{props.label}: </label>
          <select onChange={props.onChange}>
              {optionList}
          </select>
          <div id="relative settings">
              <EffectSelect options={this.props.effectData}/>
          </div>
      </div>
    );
}

function EffectSelect(props) {
    const optionList = props.options.map((option) =>
        <option key={option.toString()} value={option.toString()}>{option.toString()}</option>);
    return(
      <div>
          <label>{props.label}: </label>
          <select onChange={props.onChange}>
              {optionList}
          </select>
          <div id="relative settings">
            </div>
      </div>
    );
}


function GlobalFieldList(props) {
    const fieldList = props.fields.map((field) =>
        <Field key={field} name={field} />
    );
    return(fieldList);
}


function Field(props) {
    return(
        <div>
            {props.name}
            <input type="text" value={props.value} />
        </div>
      );
}

export default App;
